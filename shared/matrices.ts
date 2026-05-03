import { calculateStandardDeviation, divideToWhole } from "./math.ts";

export const createVector = (
  dimensionCount: number,
  fillFunction: () => number = () => 0,
) => new Array(dimensionCount).fill(0).map(fillFunction);

export const createMatrix = (
  vectorCount: number,
  dimensionCount: number,
  fillFunction?: () => number,
) => {
  return new Array(vectorCount)
    .fill(0)
    .map((_) => createVector(dimensionCount, fillFunction));
};

export const validateSize = (
  matrix: number[][],
  expectedVectorCount: number,
  expectedDepth?: number,
) => {
  if (matrix.length !== expectedVectorCount) {
    throw new Error(
      `matrix vector count (${matrix.length}) doesn't match expected vector count ${expectedVectorCount}`,
    );
  }

  const firstVector = matrix[0];

  if (!firstVector) {
    throw new Error(`matrix has no vectors`);
  }

  if (expectedDepth !== undefined && firstVector.length !== expectedDepth) {
    throw new Error(
      `m has unexpected vector depth ${firstVector.length}, expected ${expectedDepth}`,
    );
  }

  validateConsistentNestedArrayLength(matrix);
};

export const validateConsistentNestedArrayLength = (matrix: number[][]) => {
  const firstVector = matrix[0];

  if (!firstVector) {
    // empty, is valid
    return;
  }

  const vectorDepth = firstVector.length;

  for (const [index, vector] of Object.entries(matrix)) {
    if (vector.length !== vectorDepth) {
      throw new Error(
        `Vector at index ${index} has unexpected depth ${vector.length} (expected ${vectorDepth})`,
      );
    }
  }
};

export const multiplyMatrices = (
  m1: number[][],
  m2: number[][],
): number[][] => {
  const m3 = new Array(m1.length)
    .fill(0)
    .map(() => new Array(m2[0]!.length).fill(0));

  for (let i = 0; i < m1.length; i++) {
    const m1Vector = m1[i]!;
    const m3Vector = m3[i]!;
    for (let k = 0; k < m1Vector.length; k++) {
      const scalar = m1Vector[k]!;
      const m2Vector = m2[k]!;

      for (let j = 0; j < m2Vector.length; j++) {
        m3Vector[j] += scalar * m2Vector[j]!;
      }
    }
  }

  return m3;
};

export const multiplyMatrixWithVector = (
  vector: number[],
  matrix: number[][],
): number[] => {
  const multipliedMatrix = multiplyMatrices([vector], matrix);

  return multipliedMatrix[0]!;
};

export const operateOnVectors = (
  vector1: number[],
  vector2: number[],
  operation: (v1: number, v2: number) => number,
): number[] => {
  validateSize([vector1], 1, vector2.length);

  return vector1.map((value1, dimensionIndex) =>
    operation(value1, vector2[dimensionIndex]!),
  );
};

export const operateOnMatrices = (
  m1: number[][],
  m2: number[][],
  operation: (v1: number, v2: number) => number,
): number[][] => {
  validateSize(m1, m2.length, m2[0]!.length);

  return m1.map((vector1, vectorIndex) =>
    operateOnVectors(vector1, m2[vectorIndex]!, operation),
  );
};

export const operateOnMatrix = (
  m1: number[][],
  operation: (v: number) => number,
): number[][] => {
  return m1.map((vector1) => vector1.map(operation));
};

export const applyScalarToVector = (scalar: number, vector: number[]) =>
  vector.map((value) => value * scalar);

export const applyScalarToMatrix = (scalar: number, matrix: number[][]) =>
  matrix.map((vector) => applyScalarToVector(scalar, vector));

export const addVectors = (vector1: number[], vector2: number[]) =>
  operateOnVectors(vector1, vector2, (value1, value2) => value1 + value2);

export const addVectorsInMatrix = (matrix: number[][]) => {
  const vectorDimensions = matrix[0]?.length ?? 0;
  validateSize(matrix, matrix.length, vectorDimensions);

  return matrix.reduce(
    (sumVector, nextVector) => addVectors(sumVector, nextVector),
    new Array<number>(vectorDimensions).fill(0),
  );
};

export const addMatrices = (matrix1: number[][], matrix2: number[][]) =>
  operateOnMatrices(matrix1, matrix2, (value1, value2) => value1 + value2);

export const transpose = (matrix: number[][]): number[][] => {
  const vectors = matrix.length;
  const depth = matrix[0]!.length;

  return new Array(depth).fill(0).map((_, newVectorIndex) => {
    return new Array(vectors).fill(0).map((_, newDepthIndex) => {
      return matrix[newDepthIndex]![newVectorIndex]!;
    });
  });
};

export const normalize = (matrix: number[][]): number[][] => {
  return matrix.map((vector) => {
    const { average, standardDeviation } = calculateStandardDeviation(vector);

    return vector.map(
      (scalar) =>
        (scalar - average) /
        (standardDeviation +
          // to prevent 0-divisions
          Number.EPSILON),
    );
  });
};

export const getMatrixSize = (matrix: number[][]) => {
  const vectors = Object.values(matrix);

  const firstVector = vectors[0];

  return {
    vectorCount: vectors.length,
    dimensionsCount: firstVector ? firstVector.length : 0,
  };
};

export const getMatrixParameterCount = (matrix: number[][]) => {
  const size = getMatrixSize(matrix);

  return size.vectorCount * size.dimensionsCount;
};

export const concatenateMatricesVertically = (matrices: number[][][]) => {
  const vectors = matrices[0]!.length;

  return matrices.reduce(
    (partial, matrix) =>
      partial.map((vector, vectorIndex) => [
        ...vector,
        ...matrix[vectorIndex]!,
      ]),
    new Array<number[]>(vectors).fill([]),
  );
};

export const sliceRows = (matrix: number[][], start: number, end: number) => {
  return matrix.map((vector) => vector.slice(start, end));
};

export const sliceToEqualSizes = (matrix: number[][], sectionCount: number) => {
  const matrixDimensionality = matrix[0]!.length;
  const singleSectionDimensionality = divideToWhole(
    matrixDimensionality,
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
