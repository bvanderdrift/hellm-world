export const validateSize = (m: number[][], rows: number, columns?: number) => {
  if (m.length !== rows) {
    throw new Error(
      `matrix row count (${m.length}) doesn't match expected length ${rows}`,
    );
  }

  const firstRow = m[0];

  if (!firstRow) {
    throw new Error(`m is empty`);
  }

  if (columns !== undefined && firstRow.length !== columns) {
    throw new Error(
      `m has unexpected column size ${firstRow.length}, expected ${columns}`,
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

  const m1RowCount = m1.length;
  const m1ColumnsCount = m1[0]!.length;

  validateSize(m2, m1ColumnsCount);

  const m3 = new Array(m1RowCount).fill(0).map((_, rowM3) => {
    return new Array(m2ColumnCount).fill(0).map((_, columnM3) => {
      return m1[rowM3]!.reduce((sum, e, columnM1) => {
        return sum + e * m2[columnM1]![columnM3]!;
      }, 0);
    });
  });

  return m3;
};

export const flipMatrix = (matrix: number[][]): number[][] => {
  const rows = matrix.length;
  const columns = matrix[0]!.length;

  return new Array(columns).fill(0).map((_, newRowIndex) => {
    return new Array(rows).fill(0).map((_, newColumnIndex) => {
      return matrix[newColumnIndex]![newRowIndex]!;
    });
  });
};
