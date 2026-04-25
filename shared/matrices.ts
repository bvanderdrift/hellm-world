import { calculateStandardDeviation } from "./math.ts";

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

export const applyScalarToVector = (scalar: number, vector: number[]) =>
  vector.map((value) => value * scalar);

export const applyScalarToMatrix = (scalar: number, matrix: number[][]) =>
  matrix.map((vector) => applyScalarToVector(scalar, vector));

export const addVectors = (vector1: number[], vector2: number[]) => {
  if (vector1.length !== vector2.length) {
    throw new Error(
      `Vector1 size ${vector1.length} doesn't match vector2 size ${vector2.length}`,
    );
  }

  return vector1.map((e1, index) => e1 + vector2[index]!);
};

export const addVectorsInMatrix = (matrix: number[][]) => {
  const vectorDimensions = matrix[0]?.length ?? 0;
  validateSize(matrix, matrix.length, vectorDimensions);

  return matrix.reduce(
    (sumVector, nextVector) => addVectors(sumVector, nextVector),
    new Array<number>(vectorDimensions).fill(0),
  );
};

export const addMatrices = (matrix1: number[][], matrix2: number[][]) => {
  validateSize(matrix1, matrix2.length, matrix2[0]!.length);

  return matrix1.map((vector1, vector1Index) =>
    addVectors(vector1, matrix2[vector1Index]!),
  );
};

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
