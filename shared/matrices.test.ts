import { describe, expect, it } from "vitest";
import { calculateStandardDeviation } from "./math.ts";
import {
  addMatrices,
  addVectors,
  addVectorsInMatrix,
  applyScalarToMatrix,
  applyScalarToVector,
  createMatrix,
  flipMatrix,
  getMatrixSize,
  multiplyMatrixWithVector,
  multiplyMatrices,
  normalize,
  validateConsistentNestedArrayLength,
  validateSize,
} from "./matrices.ts";

describe("createMatrix", () => {
  it("creates a matrix with the requested vector and dimension counts", () => {
    expect(createMatrix(2, 3)).toEqual([
      [0, 0, 0],
      [0, 0, 0],
    ]);
  });

  it("creates independent row arrays", () => {
    const matrix = createMatrix(2, 2);

    matrix[0]![0] = 1;

    expect(matrix).toEqual([
      [1, 0],
      [0, 0],
    ]);
    expect(matrix[0]).not.toBe(matrix[1]);
  });

  it("handles zero vectors", () => {
    expect(createMatrix(0, 3)).toEqual([]);
  });

  it("handles zero dimensions", () => {
    expect(createMatrix(2, 0)).toEqual([[], []]);
  });
});

describe("getMatrixSize", () => {
  it("returns the vector count and dimension count for a matrix", () => {
    expect(
      getMatrixSize([
        [1, 2, 3],
        [4, 5, 6],
      ]),
    ).toEqual({
      vectorCount: 2,
      dimensionsCount: 3,
    });
  });

  it("returns zero dimensions for an empty matrix", () => {
    expect(getMatrixSize([])).toEqual({
      vectorCount: 0,
      dimensionsCount: 0,
    });
  });

  it("uses the first vector width as the dimension count", () => {
    expect(getMatrixSize([[], []])).toEqual({
      vectorCount: 2,
      dimensionsCount: 0,
    });
  });
});

describe("multiplyMatrices", () => {
  it("multiplies two 1x1 matrices", () => {
    expect(multiplyMatrices([[3]], [[4]])).toEqual([[12]]);
  });

  it("multiplies two 2x2 matrices", () => {
    expect(
      multiplyMatrices(
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ),
    ).toEqual([
      [19, 22],
      [43, 50],
    ]);
  });

  it("multiplies a 2x3 matrix by a 3x2 matrix", () => {
    expect(
      multiplyMatrices(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        [
          [7, 8],
          [9, 10],
          [11, 12],
        ],
      ),
    ).toEqual([
      [58, 64],
      [139, 154],
    ]);
  });

  it("leaves a matrix unchanged when multiplied by the identity matrix", () => {
    expect(
      multiplyMatrices(
        [
          [2, -1],
          [0, 3],
        ],
        [
          [1, 0],
          [0, 1],
        ],
      ),
    ).toEqual([
      [2, -1],
      [0, 3],
    ]);
  });

  it("throws when the inner dimensions do not match", () => {
    expect(() =>
      multiplyMatrices(
        [
          [1, 2],
          [3, 4],
        ],
        [[5, 6, 7]],
      ),
    ).toThrow();
  });

  it("throws when the second matrix is empty", () => {
    expect(() => multiplyMatrices([[1, 2]], [])).toThrow("m2 is empty");
  });

  it("handles 1x1 with 1x2 matrix", () => {
    const out = multiplyMatrices([[2]], [[1, 1]]);

    expect(out).toEqual([[2, 2]]);
  });
});

describe("multiplyMatrixWithVector", () => {
  it("multiplies a vector by a matrix", () => {
    expect(
      multiplyMatrixWithVector(
        [
          [1, 2],
          [3, 4],
          [5, 6],
        ],
        [7, 8, 9],
      ),
    ).toEqual([76, 100]);
  });

  it("throws when the vector length does not match the matrix row count", () => {
    expect(() =>
      multiplyMatrixWithVector(
        [
          [1, 2],
          [3, 4],
        ],
        [5],
      ),
    ).toThrow("matrix vector count (2) doesn't match expected vector count 1");
  });
});

describe("flipMatrix", () => {
  it("should flip square matrix", () => {
    const flipped = flipMatrix([
      [1, 2],
      [3, 4],
    ]);

    expect(flipped).toEqual([
      [1, 3],
      [2, 4],
    ]);
  });

  it("should flip non-square matrix", () => {
    const flipped = flipMatrix([
      [1, 2, 3],
      [4, 5, 6],
    ]);

    expect(flipped).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);
  });
});

describe("applyScalarToVector", () => {
  it("multiplies every vector value by the scalar", () => {
    expect(applyScalarToVector(3, [2, -4, 0.5])).toEqual([6, -12, 1.5]);
  });

  it("returns a new vector without mutating the input", () => {
    const vector = [1, 2, 3];

    const scaled = applyScalarToVector(2, vector);

    expect(scaled).toEqual([2, 4, 6]);
    expect(vector).toEqual([1, 2, 3]);
    expect(scaled).not.toBe(vector);
  });
});

describe("applyScalarToMatrix", () => {
  it("multiplies every matrix value by the scalar", () => {
    expect(
      applyScalarToMatrix(0.5, [
        [2, 4],
        [-6, 8],
      ]),
    ).toEqual([
      [1, 2],
      [-3, 4],
    ]);
  });

  it("returns a new matrix without mutating the input", () => {
    const matrix = [
      [1, 2],
      [3, 4],
    ];

    const scaled = applyScalarToMatrix(-1, matrix);

    expect(scaled).toEqual([
      [-1, -2],
      [-3, -4],
    ]);
    expect(matrix).toEqual([
      [1, 2],
      [3, 4],
    ]);
    expect(scaled).not.toBe(matrix);
    expect(scaled[0]).not.toBe(matrix[0]);
  });
});

describe("normalize", () => {
  it("normalizes each row to zero mean and unit standard deviation", () => {
    const normalized = normalize([
      [1, 2, 3],
      [10, 20, 30],
    ]);

    for (const row of normalized) {
      const { average, standardDeviation } = calculateStandardDeviation(row);

      expect(average).toBeCloseTo(0, 10);
      expect(standardDeviation).toBeCloseTo(1, 10);
    }
  });

  it("treats each row independently", () => {
    const normalized = normalize([
      [1, 2, 3],
      [101, 102, 103],
      [10, 20, 30],
    ]);

    const [baseRow, shiftedRow, scaledRow] = normalized;

    for (const [index, value] of (baseRow ?? []).entries()) {
      expect(value).toBeCloseTo(shiftedRow?.[index] ?? NaN, 10);
      expect(value).toBeCloseTo(scaledRow?.[index] ?? NaN, 10);
    }
  });

  it("returns a new matrix without mutating the input", () => {
    const matrix = [
      [1, 2, 3],
      [4, 5, 6],
    ];
    const original = matrix.map((row) => [...row]);

    const normalized = normalize(matrix);

    expect(matrix).toEqual(original);
    expect(normalized).not.toBe(matrix);
    expect(normalized[0]).not.toBe(matrix[0]);
    expect(normalized[1]).not.toBe(matrix[1]);
  });

  it("gracefully handles constant vectors", () => {
    const matrix = [[3, 3, 3, 3]];
    expect(normalize(matrix)).toEqual([[0, 0, 0, 0]]);
  });
});

describe("addVectors", () => {
  it("adds vectors element-wise", () => {
    expect(addVectors([1, 2, 3], [4, 5, 6])).toEqual([5, 7, 9]);
  });

  it("handles negative and decimal values", () => {
    expect(addVectors([-1.5, 2, 0], [0.5, -3, 4.25])).toEqual([-1, -1, 4.25]);
  });

  it("throws when vector sizes do not match", () => {
    expect(() => addVectors([1, 2], [3])).toThrow(
      "m has unexpected vector depth 2, expected 1",
    );
  });
});

describe("addMatrices", () => {
  it("adds matrices element-wise", () => {
    expect(
      addMatrices(
        [
          [1, 2],
          [3, 4],
        ],
        [
          [10, 20],
          [30, 40],
        ],
      ),
    ).toEqual([
      [11, 22],
      [33, 44],
    ]);
  });

  it("handles negative and decimal values", () => {
    expect(
      addMatrices(
        [
          [-1.5, 0],
          [2.25, -3],
        ],
        [
          [0.5, 4],
          [-2.25, 1.5],
        ],
      ),
    ).toEqual([
      [-1, 4],
      [0, -1.5],
    ]);
  });

  it("throws when matrix sizes do not match", () => {
    expect(() =>
      addMatrices(
        [
          [1, 2],
          [3, 4],
        ],
        [[5, 6]],
      ),
    ).toThrow("matrix vector count (2) doesn't match expected vector count 1");
  });

  it("throws when row widths do not match", () => {
    expect(() => addMatrices([[1, 2]], [[3, 4, 5]])).toThrow(
      "m has unexpected vector depth 2, expected 3",
    );
  });
});

describe("addVectorsInMatrix", () => {
  it("sums all rows into one vector", () => {
    expect(
      addVectorsInMatrix([
        [1, 2, 3],
        [4, 5, 6],
        [-2, 10, 0],
      ]),
    ).toEqual([3, 17, 9]);
  });

  it("returns a zero vector for an empty row", () => {
    expect(addVectorsInMatrix([[], []])).toEqual([]);
  });

  it("throws when matrix rows have inconsistent widths", () => {
    expect(() => addVectorsInMatrix([[1, 2], [3]])).toThrow(
      "Vector at index 1 has unexpected depth 1 (expected 2)",
    );
  });
});

describe("validateSize", () => {
  it("accepts matrices with the expected row count and depth", () => {
    expect(() =>
      validateSize(
        [
          [1, 2],
          [3, 4],
        ],
        2,
        2,
      ),
    ).not.toThrow();
  });

  it("throws when the row count is wrong", () => {
    expect(() => validateSize([[1, 2]], 2, 2)).toThrow(
      "matrix vector count (1) doesn't match expected vector count 2",
    );
  });

  it("throws when the first row depth is wrong", () => {
    expect(() => validateSize([[1, 2, 3]], 1, 2)).toThrow(
      "m has unexpected vector depth 3, expected 2",
    );
  });

  it("throws when the matrix is empty", () => {
    expect(() => validateSize([], 0)).toThrow("matrix has no vectors");
  });
});

describe("validateConsistentNestedArrayLength", () => {
  it("accepts an empty matrix", () => {
    expect(() => validateConsistentNestedArrayLength([])).not.toThrow();
  });

  it("accepts rows with matching depths", () => {
    expect(() =>
      validateConsistentNestedArrayLength([
        [1, 2],
        [3, 4],
      ]),
    ).not.toThrow();
  });

  it("throws when a later row has a different depth", () => {
    expect(() =>
      validateConsistentNestedArrayLength([
        [1, 2],
        [3, 4, 5],
      ]),
    ).toThrow("Vector at index 1 has unexpected depth 3 (expected 2)");
  });
});
