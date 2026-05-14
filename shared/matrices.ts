import { calculateStandardDeviation, divideToWhole } from "./math.ts";

export const createMatrix = (
  vectorCount: number,
  dimensionCount: number,
  fillFunction?: () => number,
): Matrix => {
  let values = new Float32Array(vectorCount * dimensionCount).fill(0);

  if (fillFunction) {
    values = values.map(fillFunction);
  }

  return {
    values,
    vectors: vectorCount,
    dimensions: dimensionCount,
  };
};

export type Matrix = {
  vectors: number;
  dimensions: number;
  values: Float32Array;
};

export const getFlatIndex = (i: number, j: number, dimensionality: number) =>
  i * dimensionality + j;

export const multiplyMatrices = (m1: Matrix, m2: Matrix): Matrix => {
  const m3: Matrix = {
    values: new Float32Array(m1.vectors * m2.dimensions).fill(0),
    vectors: m1.vectors,
    dimensions: m2.dimensions,
  };

  for (let i = 0; i < m1.vectors; i++) {
    for (let k = 0; k < m1.dimensions; k++) {
      const m1Index = getFlatIndex(i, k, m1.dimensions);
      for (let j = 0; j < m2.dimensions; j++) {
        m3.values[getFlatIndex(i, j, m2.dimensions)]! +=
          m1.values[m1Index]! * m2.values[getFlatIndex(k, j, m2.dimensions)]!;
      }
    }
  }

  return m3;
};

export const addVectorAcrossMatrix = (v: Matrix, m: Matrix): Matrix => {
  const output = createMatrix(m.vectors, m.dimensions);

  for (let i = 0; i < m.vectors; i++) {
    for (let j = 0; j < m.dimensions; j++) {
      const matrixIndexFlat = getFlatIndex(i, j, output.dimensions);
      output.values[matrixIndexFlat]! =
        m.values[matrixIndexFlat]! + v.values[j]!;
    }
  }

  return output;
};

export const operateOnMatrices = (
  m1: Matrix,
  m2: Matrix,
  operation: (v1: number, v2: number) => number,
): Matrix => {
  return {
    ...m1,
    values: m1.values.map((value1, valueIndex) =>
      operation(value1, m2.values[valueIndex]!),
    ),
  };
};

export const operateOnMatrix = (
  m1: Matrix,
  operation: (v: number) => number,
): Matrix => {
  return {
    ...m1,
    values: m1.values.map(operation),
  };
};

export const applyScalarToMatrix = (scalar: number, matrix: Matrix) =>
  operateOnMatrix(matrix, (v) => v * scalar);

export const addVectorsInMatrix = (matrix: Matrix) => {
  const outputVector = createMatrix(1, matrix.dimensions);

  for (let i = 0; i < matrix.vectors; i++) {
    for (let j = 0; j < matrix.dimensions; j++) {
      outputVector.values[j]! +=
        matrix.values[getFlatIndex(i, j, matrix.dimensions)]!;
    }
  }

  return outputVector;
};

export const addMatrices = (matrix1: Matrix, matrix2: Matrix) =>
  operateOnMatrices(matrix1, matrix2, (value1, value2) => value1 + value2);

export const transpose = (matrix: Matrix): Matrix => {
  const output = createMatrix(matrix.dimensions, matrix.vectors);

  for (let i = 0; i < matrix.vectors; i++) {
    for (let j = 0; j < matrix.dimensions; j++) {
      output.values[getFlatIndex(j, i, output.dimensions)]! =
        matrix.values[getFlatIndex(i, j, matrix.dimensions)]!;
    }
  }

  return output;
};

export const getRawVector = (matrix: Matrix, vectorIndex: number) =>
  matrix.values.slice(
    vectorIndex * matrix.dimensions,
    (vectorIndex + 1) * matrix.dimensions,
  );

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

export const getMatrixParameterCount = (matrix: Matrix) => {
  return matrix.vectors * matrix.dimensions;
};

export const concatenateMatricesVertically = (matrices: Matrix[]) => {
  const baseVectors = matrices[0]!.vectors;
  const baseDimensions = matrices[0]!.dimensions;

  const output = createMatrix(baseVectors, baseDimensions * matrices.length);

  for (let m = 0; m < matrices.length; m++) {
    const matrix = matrices[m]!;

    if (matrix.vectors !== baseVectors) {
      throw new Error(
        `Mismatching vector count ${matrix.vectors} - ${baseVectors}`,
      );
    }
    if (matrix.dimensions !== baseDimensions) {
      throw new Error(
        `Mismatching dimension count ${matrix.dimensions} - ${baseDimensions}`,
      );
    }

    for (let i = 0; i < matrix.vectors; i++) {
      for (let j = 0; j < matrix.dimensions; j++) {
        const outputIndex = getFlatIndex(
          i,
          m * matrix.dimensions + j,
          output.dimensions,
        );

        output.values[outputIndex] =
          matrix.values[getFlatIndex(i, j, matrix.dimensions)]!;
      }
    }
  }

  return output;
};

export const sliceRows = (matrix: Matrix, start: number, end: number) => {
  const output = createMatrix(matrix.vectors, end - start);

  for (let i = 0; i < matrix.vectors; i++) {
    for (let j = start; j < end; j++) {
      output.values[getFlatIndex(i, j - start, output.dimensions)]! =
        matrix.values[getFlatIndex(i, j, matrix.dimensions)]!;
    }
  }

  return output;
};

export const sliceToEqualSizes = (matrix: Matrix, sectionCount: number) => {
  const singleSectionDimensionality = divideToWhole(
    matrix.dimensions,
    sectionCount,
  );

  return new Array(sectionCount).fill(0).map((_, sectionIndex) => {
    return sliceRows(
      matrix,
      sectionIndex * singleSectionDimensionality,
      (sectionIndex + 1) * singleSectionDimensionality,
    );
  });
};
