import { describe, expect, it } from "vitest";
import { getFlatIndex, type Matrix } from "../shared/matrices.ts";
import { getPositionEncoding } from "./position-encoding.ts";
import { TESTING_PRECISION } from "../testing/constants.ts";

const getValue = (matrix: Matrix, rowIndex: number, columnIndex: number) => {
  const index = getFlatIndex(rowIndex, columnIndex, matrix.dimensions);
  const value = matrix.values[index];

  if (value === undefined) {
    throw new Error(`Missing value at [${rowIndex}, ${columnIndex}]`);
  }

  return value;
};

const getVector = (matrix: Matrix, rowIndex: number) =>
  Array.from(
    matrix.values.slice(
      rowIndex * matrix.dimensions,
      (rowIndex + 1) * matrix.dimensions,
    ),
  );

describe("getPositionEncoding", () => {
  it("returns one row per token and one column per dimension", () => {
    const encoding = getPositionEncoding(3, 4);

    expect(encoding.vectors).toBe(3);
    expect(encoding.dimensions).toBe(4);
  });

  it("uses sin for even dimensions and cos for odd dimensions at position 0", () => {
    const encoding = getPositionEncoding(1, 6);

    expect(getVector(encoding, 0)).toEqual([0, 1, 0, 1, 0, 1]);
  });

  it("uses the same angle for each even and odd dimension pair", () => {
    const encoding = getPositionEncoding(2, 6);

    for (let featureIndex = 0; featureIndex < 6; featureIndex += 2) {
      const sinValue = getValue(encoding, 1, featureIndex);
      const cosValue = getValue(encoding, 1, featureIndex + 1);

      expect(sinValue ** 2 + cosValue ** 2).toBeCloseTo(1, TESTING_PRECISION);
    }
  });

  it("matches the sinusoidal formula for the first two dimension pairs", () => {
    const encoding = getPositionEncoding(2, 4);

    expect(getValue(encoding, 1, 0)).toBeCloseTo(
      Math.sin(1),
      TESTING_PRECISION,
    );
    expect(getValue(encoding, 1, 1)).toBeCloseTo(
      Math.cos(1),
      TESTING_PRECISION,
    );
    expect(getValue(encoding, 1, 2)).toBeCloseTo(
      Math.sin(0.01),
      TESTING_PRECISION,
    );
    expect(getValue(encoding, 1, 3)).toBeCloseTo(
      Math.cos(0.01),
      TESTING_PRECISION,
    );
  });
});
