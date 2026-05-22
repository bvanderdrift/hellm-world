import { expect } from "vitest";
import { createMatrix, getFlatIndex, type Matrix } from "../shared/matrices.ts";
import { TESTING_PRECISION } from "./constants.ts";

export const matrixFrom = (rows: number[][]): Matrix => {
  const vectors = rows.length;
  const dimensions = rows[0]!.length;
  const m = createMatrix(vectors, dimensions);
  for (let i = 0; i < vectors; i++) {
    for (let j = 0; j < dimensions; j++) {
      m.values[getFlatIndex(i, j, dimensions)] = rows[i]![j]!;
    }
  }
  return m;
};

export const expectMatrixCloseTo = (
  actual: Matrix,
  expected: Matrix,
  precision = TESTING_PRECISION,
) => {
  expect(actual.vectors).toBe(expected.vectors);
  expect(actual.dimensions).toBe(expected.dimensions);

  for (let i = 0; i < expected.values.length; i++) {
    expect(actual.values[i]).toBeCloseTo(expected.values[i]!, precision);
  }
};
