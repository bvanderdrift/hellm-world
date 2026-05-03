import { describe, expect, it } from "vitest";
import { softmax } from "../../shared/math.ts";
import { calculateLoss } from "../calculateLoss.ts";
import { probabilityOutputBackprop } from "./probabilityOutputBackprop.ts";

describe("probabilityOutputBackprop", () => {
  it("places cross-entropy logit gradients only on the last context position", () => {
    const logits = [
      [9, 8, 7, 6],
      [3, 2, 1, 0],
      [0.7, -1.1, 2.2, -0.4],
    ];
    const outputProbabilities = softmax(logits[2]!);
    const correctTokenIndex = 3;

    const gradients = probabilityOutputBackprop(
      logits,
      outputProbabilities,
      logits.length,
      correctTokenIndex,
    );

    expect(gradients[0]).toEqual([0, 0, 0, 0]);
    expect(gradients[1]).toEqual([0, 0, 0, 0]);
    expect(gradients[2]).toEqual(
      outputProbabilities.map(
        (probability, vocabIndex) =>
          probability - (vocabIndex === correctTokenIndex ? 1 : 0),
      ),
    );
  });

  it("treats the correct token index as a vocabulary index", () => {
    const logits = [
      [0, 0, 0, 0],
      [0.7, -1.1, 2.2, -0.4],
    ];
    const outputProbabilities = softmax(logits[1]!);
    const correctTokenIndex = 3;

    const gradients = probabilityOutputBackprop(
      logits,
      outputProbabilities,
      logits.length,
      correctTokenIndex,
    );

    expect(gradients[1]![1]).toBeCloseTo(outputProbabilities[1]!, 10);
    expect(gradients[1]![3]).toBeCloseTo(outputProbabilities[3]! - 1, 10);
  });

  it("matches finite differences of calculateLoss for trained-position logits", () => {
    const logits = [
      [10, -10, 5],
      [1.2, -0.7, 0.3],
    ];
    const vocabulary = ["alpha", "beta", "gamma"];
    const correctTokenIndex = 2;
    const outputProbabilities = softmax(logits[1]!);
    const epsilon = 0.000001;

    const gradients = probabilityOutputBackprop(
      logits,
      outputProbabilities,
      logits.length,
      correctTokenIndex,
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
