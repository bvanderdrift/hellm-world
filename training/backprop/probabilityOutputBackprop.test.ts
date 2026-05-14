import { describe, expect, it } from "vitest";
import { softmax } from "../../shared/math.ts";
import {
  createMatrix,
  getFlatIndex,
  getRawVector,
  type Matrix,
} from "../../shared/matrices.ts";
import { probabilityOutputBackprop } from "./probabilityOutputBackprop.ts";

const m = (data: number[][]): Matrix => {
  const mat = createMatrix(data.length, data[0]!.length);
  mat.values.set(data.flat());
  return mat;
};

const buildProbabilities = (logitsData: number[][]): Matrix => {
  const rows = logitsData.map((row) => Array.from(softmax(new Float32Array(row))));
  return m(rows);
};

describe("probabilityOutputBackprop", () => {
  it("places cross-entropy logit gradients on every trained context position", () => {
    const logitsData = [
      [9, 8, 7, 6],
      [3, 2, 1, 0],
      [0.7, -1.1, 2.2, -0.4],
    ];
    const logits = m(logitsData);
    const outputProbabilities = buildProbabilities(logitsData);
    const correctTokenIndices = [1, 2, 3];

    const gradients = probabilityOutputBackprop(
      logits,
      outputProbabilities,
      correctTokenIndices,
    );

    for (let row = 0; row < 3; row++) {
      const probRow = getRawVector(outputProbabilities, row);
      const gradRow = getRawVector(gradients, row);
      for (let v = 0; v < probRow.length; v++) {
        expect(gradRow[v]).toBeCloseTo(
          probRow[v]! - (v === correctTokenIndices[row] ? 1 : 0),
          5,
        );
      }
    }
  });

  it("treats the correct token index as a vocabulary index", () => {
    const logitsData = [
      [0, 0, 0, 0],
      [0.7, -1.1, 2.2, -0.4],
    ];
    const logits = m(logitsData);
    const outputProbabilities = buildProbabilities(logitsData);
    const correctTokenIndex = 3;

    const gradients = probabilityOutputBackprop(
      logits,
      outputProbabilities,
      [0, correctTokenIndex],
    );

    const prob1 = getRawVector(outputProbabilities, 1);
    const grad1 = getRawVector(gradients, 1);
    expect(grad1[1]).toBeCloseTo(prob1[1]!, 5);
    expect(grad1[3]).toBeCloseTo(prob1[3]! - 1, 5);
  });

  it("matches finite differences of cross-entropy loss for trained-position logits", () => {
    const logitsData = [
      [10, -10, 5],
      [1.2, -0.7, 0.3],
    ];
    const logits = m(logitsData);
    const outputProbabilities = buildProbabilities(logitsData);
    const correctTokenIndex = 2;
    const epsilon = 0.000001;

    const loss = (l: Float32Array) =>
      -Math.log(softmax(l)[correctTokenIndex]!);

    const gradients = probabilityOutputBackprop(
      logits,
      outputProbabilities,
      [0, correctTokenIndex],
    );

    const row1 = logitsData[1]!;
    for (let vocabIndex = 0; vocabIndex < row1.length; vocabIndex++) {
      const increasedLogits = new Float32Array(row1);
      const decreasedLogits = new Float32Array(row1);

      increasedLogits[vocabIndex]! += epsilon;
      decreasedLogits[vocabIndex]! -= epsilon;

      const numericalGradient =
        (loss(increasedLogits) - loss(decreasedLogits)) / (2 * epsilon);

      expect(
        gradients.values[getFlatIndex(1, vocabIndex, gradients.dimensions)],
      ).toBeCloseTo(numericalGradient, 1);
    }
  });
});
