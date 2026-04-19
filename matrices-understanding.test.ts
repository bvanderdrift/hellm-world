// Special test file that doesn't test implementation, but is for me to rapidly test edge-cases to understand matrix math

import { describe, expect, it } from "vitest";
import { addMatrices, multiplyMatrices } from "./matrices.ts";

describe("summing vs concatenation", () => {
  it("is true that M1 @ W1 + M2 @ W2 == Concat_Dimensions(M1, M2) @ Concat_Vectors(W1, W2)", () => {
    const m1 = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ];

    const m2 = [
      [9, 10, 11, 12],
      [13, 14, 15, 16],
    ];

    const w1 = [
      [17, 18, 19],
      [20, 30, 40],
      [50, 60, 70],
      [80, 90, 22],
    ];

    const w2 = [
      [33, 44, 55],
      [66, 77, 88],
      [99, 23, 24],
      [25, 26, 27],
    ];

    const sum = addMatrices(multiplyMatrices(m1, w1), multiplyMatrices(m2, w2));
    
    const concat = multiplyMatrices(
      m1.map((m1Vector, m1ColumnNumber) => [
        ...m1Vector,
        ...m2[m1ColumnNumber]!,
      ]),
      [...w1, ...w2],
    );

    expect(sum).toEqual(concat);
  });
});
