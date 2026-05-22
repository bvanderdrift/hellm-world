// Special test file that doesn't test implementation, but is for me to rapidly test edge-cases to understand matrix math

import { describe, expect, it } from "vitest";
import { addMatrices, createMatrix, multiplyMatrices } from "./matrices.ts";
import { matrixFrom } from "../testing/testing-utils.ts";

describe("summing vs concatenation", () => {
  it("is true that M1 @ W1 + M2 @ W2 == Concat_Dimensions(M1, M2) @ Concat_Vectors(W1, W2)", () => {
    const m1 = matrixFrom([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ]);

    const m2 = matrixFrom([
      [9, 10, 11, 12],
      [13, 14, 15, 16],
    ]);

    const w1 = matrixFrom([
      [17, 18, 19],
      [20, 30, 40],
      [50, 60, 70],
      [80, 90, 22],
    ]);

    const w2 = matrixFrom([
      [33, 44, 55],
      [66, 77, 88],
      [99, 23, 24],
      [25, 26, 27],
    ]);

    const sum = addMatrices(multiplyMatrices(m1, w1), multiplyMatrices(m2, w2));

    const concatDimensions = createMatrix(m1.vectors, m1.dimensions + m2.dimensions);
    for (let i = 0; i < m1.vectors; i++) {
      for (let j = 0; j < m1.dimensions; j++) {
        concatDimensions.values[i * concatDimensions.dimensions + j] = m1.values[i * m1.dimensions + j]!;
      }
      for (let j = 0; j < m2.dimensions; j++) {
        concatDimensions.values[i * concatDimensions.dimensions + m1.dimensions + j] = m2.values[i * m2.dimensions + j]!;
      }
    }

    const concatVectors = createMatrix(w1.vectors + w2.vectors, w1.dimensions);
    concatVectors.values.set(w1.values);
    concatVectors.values.set(w2.values, w1.values.length);

    const concat = multiplyMatrices(concatDimensions, concatVectors);

    expect(sum).toEqual(concat);
  });
});
