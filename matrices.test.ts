import { describe, expect, it } from "vitest";
import {
  addMatrices,
  addVectors,
  flipMatrix,
  multiplyMatrices,
} from "./matrices.ts";

describe("multiplyMatrices", () => {
  it("multiplies two 1x1 matrices", () => {
    expect(multiplyMatrices([[3]], [[4]])).toEqual([[12]]);
  });

  it("multiplies two 2x2 matrices", () => {
    expect(
      multiplyMatrices(
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ),
    ).toEqual([
      [19, 22],
      [43, 50],
    ]);
  });

  it("multiplies a 2x3 matrix by a 3x2 matrix", () => {
    expect(
      multiplyMatrices(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        [
          [7, 8],
          [9, 10],
          [11, 12],
        ],
      ),
    ).toEqual([
      [58, 64],
      [139, 154],
    ]);
  });

  it("leaves a matrix unchanged when multiplied by the identity matrix", () => {
    expect(
      multiplyMatrices(
        [
          [2, -1],
          [0, 3],
        ],
        [
          [1, 0],
          [0, 1],
        ],
      ),
    ).toEqual([
      [2, -1],
      [0, 3],
    ]);
  });

  it("throws when the inner dimensions do not match", () => {
    expect(() =>
      multiplyMatrices(
        [
          [1, 2],
          [3, 4],
        ],
        [[5, 6, 7]],
      ),
    ).toThrow();
  });

  it("throws when the second matrix is empty", () => {
    expect(() => multiplyMatrices([[1, 2]], [])).toThrow("m2 is empty");
  });

  it("handles 1x1 with 1x2 matrix", () => {
    const out = multiplyMatrices([[2]], [[1, 1]]);

    expect(out).toEqual([[2, 2]]);
  });
});

describe("flipMatrix", () => {
  it("should flip square matrix", () => {
    const flipped = flipMatrix([
      [1, 2],
      [3, 4],
    ]);

    expect(flipped).toEqual([
      [1, 3],
      [2, 4],
    ]);
  });

  it("should flip non-square matrix", () => {
    const flipped = flipMatrix([
      [1, 2, 3],
      [4, 5, 6],
    ]);

    expect(flipped).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);
  });
});

describe("addVectors", () => {
  it("adds vectors element-wise", () => {
    expect(addVectors([1, 2, 3], [4, 5, 6])).toEqual([5, 7, 9]);
  });

  it("handles negative and decimal values", () => {
    expect(addVectors([-1.5, 2, 0], [0.5, -3, 4.25])).toEqual([-1, -1, 4.25]);
  });

  it("throws when vector sizes do not match", () => {
    expect(() => addVectors([1, 2], [3])).toThrow(
      "Vector1 size 2 doesn't match vector2 size 1",
    );
  });
});

describe("addMatrices", () => {
  it("adds matrices element-wise", () => {
    expect(
      addMatrices(
        [
          [1, 2],
          [3, 4],
        ],
        [
          [10, 20],
          [30, 40],
        ],
      ),
    ).toEqual([
      [11, 22],
      [33, 44],
    ]);
  });

  it("handles negative and decimal values", () => {
    expect(
      addMatrices(
        [
          [-1.5, 0],
          [2.25, -3],
        ],
        [
          [0.5, 4],
          [-2.25, 1.5],
        ],
      ),
    ).toEqual([
      [-1, 4],
      [0, -1.5],
    ]);
  });

  it("throws when matrix sizes do not match", () => {
    expect(() =>
      addMatrices(
        [
          [1, 2],
          [3, 4],
        ],
        [[5, 6]],
      ),
    ).toThrow("matrix vector count (2) doesn't match expected vector count 1");
  });

  it("throws when row widths do not match", () => {
    expect(() =>
      addMatrices(
        [[1, 2]],
        [[3, 4, 5]],
      ),
    ).toThrow("m has unexpected vector depth 2, expected 3");
  });
});
