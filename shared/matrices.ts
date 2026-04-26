import { calculateStandardDeviation } from "./math.ts";

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
  const m2FirstVector = m2[0];

  if (!m2FirstVector) {
    throw new Error(`m2 is empty`);
  }

  const m2DepthCount = m2FirstVector.length;

  const m1VectorCount = m1.length;
  const m1DepthCount = m1[0]!.length;

  validateSize(m2, m1DepthCount);

  const m3 = new Array(m1VectorCount).fill(0).map((_, vectorIndexM3) => {
    return new Array(m2DepthCount).fill(0).map((_, depthIndexM3) => {
      return m1[vectorIndexM3]!.reduce((sum, e, vectorIndexM1) => {
        return sum + e * m2[vectorIndexM1]![depthIndexM3]!;
      }, 0);
    });
  });

  return m3;
};

export const multiplyMatrixWithVector = (
  matrix: number[][],
  vector: number[],
): number[] => {
  const multipliedMatrix = multiplyMatrices([vector], matrix);

  validateSize(multipliedMatrix, 1);

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

export const flipMatrix = (matrix: number[][]): number[][] => {
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
      (scalar) => (scalar - average) / (standardDeviation + Number.EPSILON),
    ); // to prevent 0-divisions
  });
};
