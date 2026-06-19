import { calculateStandardDeviation } from "../math.ts";
import { type Matrix, createMatrix, getFlatIndex } from "../matrices/matrices.ts";

export const normalize = (matrix: Matrix): Matrix => {
  const output = createMatrix(matrix.vectors, matrix.dimensions);

  for (let i = 0; i < matrix.vectors; i++) {
    const { average, standardDeviation } = calculateStandardDeviation(
      matrix.values.slice(i * matrix.dimensions, (i + 1) * matrix.dimensions),
    );

    for (let j = 0; j < matrix.dimensions; j++) {
      const valueIndex = getFlatIndex(i, j, output.dimensions);
      output.values[valueIndex] =
        (matrix.values[valueIndex]! - average) /
        (standardDeviation +
          // to prevent 0-divisions
          Number.EPSILON);
    }
  }

  return output;
};
