import { expect } from "vitest";
import { createMatrix, getFlatIndex, type Matrix } from "../shared/matrices/matrices.ts";
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

export const vectorFrom = (values: number[]): Matrix => matrixFrom([values]);

/**
 * Re-lays a per-head-blocked matrix from one head-block stride to another.
 * The CPU attention reference packs each head's columns back-to-back with a
 * stride of `srcHeadStride` (its vector count); the GPU kernels stride each
 * head block by `dstHeadStride` (MAX_CONTEXT). This copies the meaningful
 * `srcHeadStride` columns of every head into the wider GPU layout, leaving the
 * padding columns zeroed.
 */
export const restripeHeadBlocks = (
  source: Matrix,
  headsCount: number,
  srcHeadStride: number,
  dstHeadStride: number,
): Matrix => {
  const out = createMatrix(source.vectors, dstHeadStride * headsCount);
  for (let i = 0; i < source.vectors; i++) {
    for (let h = 0; h < headsCount; h++) {
      for (let l = 0; l < srcHeadStride; l++) {
        out.values[getFlatIndex(i, h * dstHeadStride + l, out.dimensions)] =
          source.values[
            getFlatIndex(i, h * srcHeadStride + l, source.dimensions)
          ]!;
      }
    }
  }
  return out;
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
