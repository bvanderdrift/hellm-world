export const validateSize = (m: number[][], x: number) => {
  if (m.length !== x) {
    throw new Error(
      `matrix length (${m.length}) doesn't match expected length ${x}`,
    );
  }

  validateConsistentNestedArrayLength(m);
};

export const validateConsistentNestedArrayLength = (m: number[][]) => {
  const firstVector = m[0];

  if (!firstVector) {
    // empty, is valid
    return;
  }

  const baseLength = firstVector.length;

  for (const [index, vector] of Object.entries(m)) {
    if (vector.length !== baseLength) {
      throw new Error(
        `Vector at index ${index} has unexpected length ${vector.length} (expected ${baseLength})`,
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

  const m2ColumnCount = m2FirstVector.length;

  validateSize(m1, m2ColumnCount);

  const m1RowCount = m1.length;

  return new Array(m1RowCount).fill(0).map((_, rowM3) => {
    return new Array(m2ColumnCount).fill(0).map((_, columnM3) => {
      return m1[rowM3]!.reduce((sum, e, columnM1) => {
        return sum + e * m2[columnM1]![columnM3]!;
      }, 0);
    });
  });
};
