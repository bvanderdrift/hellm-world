import { describe, it } from "vitest";
import { embeddingsBackprop } from "./embeddingBackprop.ts";
import { matrixFrom, expectMatrixCloseTo } from "../../testing/testing-utils.ts";

describe("embeddingsBackprop", () => {
  it("scales each output gradient by the embedding width square root", () => {
    const gradients = embeddingsBackprop(
      matrixFrom([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ]),
      matrixFrom([[1, 2, 3, 4]]),
      [1],
    );

    expectMatrixCloseTo(gradients, matrixFrom([
      [0, 0, 0, 0],
      [2, 4, 6, 8],
    ]));
  });

  it("sums scaled gradients when the same token appears multiple times", () => {
    const gradients = embeddingsBackprop(
      matrixFrom([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ]),
      matrixFrom([
        [1, 2, 3, 4],
        [10, 20, 30, 40],
      ]),
      [0, 0],
    );

    expectMatrixCloseTo(gradients, matrixFrom([
      [22, 44, 66, 88],
      [0, 0, 0, 0],
    ]));
  });

  it("keeps separate embedding rows independent", () => {
    const gradients = embeddingsBackprop(
      matrixFrom([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ]),
      matrixFrom([
        [1, 2, 3, 4],
        [10, 20, 30, 40],
        [100, 200, 300, 400],
      ]),
      [2, 0, 2],
    );

    expectMatrixCloseTo(gradients, matrixFrom([
      [20, 40, 60, 80],
      [0, 0, 0, 0],
      [202, 404, 606, 808],
    ]));
  });
});
