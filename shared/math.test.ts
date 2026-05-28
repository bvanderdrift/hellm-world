import { describe, expect, it } from "vitest";
import {
  calculateStandardDeviation,
  dotProduct,
  mean,
  relu,
  sum,
} from "./math.ts";
import { TESTING_PRECISION } from "../testing/constants.ts";
import { expectMatrixCloseTo, matrixFrom } from "../testing/testing-utils.ts";

describe("sum", () => {
  it("adds numbers together", () => {
    const values = new Float32Array([1, 6, -3, 0, 2.5, -3.7]);

    expect(sum(values)).toBeCloseTo(2.8, TESTING_PRECISION);
  });
});

describe("dotProduct", () => {
  it("multiplies matching vector entries and sums them", () => {
    expect(
      dotProduct(new Float32Array([1, 2, 3]), new Float32Array([4, 5, 6])),
    ).toBe(32);
  });

  it("throws when vector sizes do not match", () => {
    expect(() =>
      dotProduct(new Float32Array([1, 2]), new Float32Array([3])),
    ).toThrow("not overlapping");
  });
});

describe("relu", () => {
  it("limits each matrix value at or above 0", () => {
    const output = relu(
      matrixFrom([
        [-1, 5, -33.2],
        [0, 12, -0.5],
      ]),
    );

    expectMatrixCloseTo(output, matrixFrom([
      [0, 5, 0],
      [0, 12, 0],
    ]));
  });
});

describe("mean", () => {
  it("averages out a set of numbers", () => {
    const values = new Float32Array([-1, 5, -33.2, 0, 12]);

    expect(mean(values)).toBeCloseTo(-3.44);
  });
});

describe("calculateStandardDeviation", () => {
  it("returns the average and population standard deviation", () => {
    const result = calculateStandardDeviation(
      new Float32Array([2, 4, 4, 4, 5, 5, 7, 9]),
    );

    expect(result.average).toBe(5);
    expect(result.standardDeviation).toBe(2);
  });

  it("returns zero when every value is identical", () => {
    const result = calculateStandardDeviation(new Float32Array([3, 3, 3, 3]));

    expect(result.average).toBe(3);
    expect(result.standardDeviation).toBe(0);
  });
});
