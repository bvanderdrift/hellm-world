import { describe, expect, it } from "vitest";
import { calculateStandardDeviation } from "./math.ts";
import {
  addMatrices,
  addVectorsInMatrix,
  applyScalarToMatrix,
  concatenateMatricesVertically,
  createMatrix,
  getFlatIndex,
  getMatrixParameterCount,
  transpose,
  multiplyMatrices,
  normalize,
  sliceRows,
  sliceToEqualSizes,
} from "./matrices.ts";
import {
  expectMatrixCloseTo,
  matrixFrom,
} from "../testing/testing-utils.ts";

describe("createMatrix", () => {
  it("creates a matrix with the requested vector and dimension counts", () => {
    const m = createMatrix(2, 3);

    expect(m.vectors).toBe(2);
    expect(m.dimensions).toBe(3);
    expect(m.values).toEqual(new Float32Array(6).fill(0));
  });

  it("creates a zero-filled flat array", () => {
    const m = createMatrix(2, 2);

    m.values[0] = 1;

    expect(m.values[0]).toBe(1);
    expect(m.values[1]).toBe(0);
    expect(m.values[2]).toBe(0);
    expect(m.values[3]).toBe(0);
  });

  it("handles zero vectors", () => {
    const m = createMatrix(0, 3);
    expect(m.vectors).toBe(0);
    expect(m.values.length).toBe(0);
  });

  it("handles zero dimensions", () => {
    const m = createMatrix(2, 0);
    expect(m.vectors).toBe(2);
    expect(m.dimensions).toBe(0);
    expect(m.values.length).toBe(0);
  });
});

describe("getMatrixParameterCount", () => {
  it("counts every scalar value in a rectangular matrix", () => {
    expect(
      getMatrixParameterCount(
        matrixFrom([
          [1, 2, 3],
          [4, 5, 6],
        ]),
      ),
    ).toBe(6);
  });

  it("returns zero for an empty matrix", () => {
    expect(getMatrixParameterCount(createMatrix(0, 0))).toBe(0);
  });
});

describe("multiplyMatrices", () => {
  it("multiplies two 1x1 matrices", () => {
    expectMatrixCloseTo(
      multiplyMatrices(matrixFrom([[3]]), matrixFrom([[4]])),
      matrixFrom([[12]]),
    );
  });

  it("multiplies two 2x2 matrices", () => {
    expectMatrixCloseTo(
      multiplyMatrices(
        matrixFrom([
          [1, 2],
          [3, 4],
        ]),
        matrixFrom([
          [5, 6],
          [7, 8],
        ]),
      ),
      matrixFrom([
        [19, 22],
        [43, 50],
      ]),
    );
  });

  it("multiplies a 2x3 matrix by a 3x2 matrix", () => {
    expectMatrixCloseTo(
      multiplyMatrices(
        matrixFrom([
          [1, 2, 3],
          [4, 5, 6],
        ]),
        matrixFrom([
          [7, 8],
          [9, 10],
          [11, 12],
        ]),
      ),
      matrixFrom([
        [58, 64],
        [139, 154],
      ]),
    );
  });

  it("leaves a matrix unchanged when multiplied by the identity matrix", () => {
    expectMatrixCloseTo(
      multiplyMatrices(
        matrixFrom([
          [2, -1],
          [0, 3],
        ]),
        matrixFrom([
          [1, 0],
          [0, 1],
        ]),
      ),
      matrixFrom([
        [2, -1],
        [0, 3],
      ]),
    );
  });

  it("handles 1x1 with 1x2 matrix", () => {
    expectMatrixCloseTo(
      multiplyMatrices(matrixFrom([[2]]), matrixFrom([[1, 1]])),
      matrixFrom([[2, 2]]),
    );
  });
});

describe("transpose", () => {
  it("should flip square matrix", () => {
    expectMatrixCloseTo(
      transpose(
        matrixFrom([
          [1, 2],
          [3, 4],
        ]),
      ),
      matrixFrom([
        [1, 3],
        [2, 4],
      ]),
    );
  });

  it("should flip non-square matrix", () => {
    expectMatrixCloseTo(
      transpose(
        matrixFrom([
          [1, 2, 3],
          [4, 5, 6],
        ]),
      ),
      matrixFrom([
        [1, 4],
        [2, 5],
        [3, 6],
      ]),
    );
  });
});

describe("applyScalarToMatrix", () => {
  it("multiplies every matrix value by the scalar", () => {
    expectMatrixCloseTo(
      applyScalarToMatrix(
        0.5,
        matrixFrom([
          [2, 4],
          [-6, 8],
        ]),
      ),
      matrixFrom([
        [1, 2],
        [-3, 4],
      ]),
    );
  });

  it("returns a new matrix without mutating the input", () => {
    const matrix = matrixFrom([
      [1, 2],
      [3, 4],
    ]);
    const originalValues = new Float32Array(matrix.values);

    const scaled = applyScalarToMatrix(-1, matrix);

    expectMatrixCloseTo(scaled, matrixFrom([
      [-1, -2],
      [-3, -4],
    ]));
    expect(matrix.values).toEqual(originalValues);
    expect(scaled.values).not.toBe(matrix.values);
  });
});

describe("normalize", () => {
  it("normalizes each row to zero mean and unit standard deviation", () => {
    const normalized = normalize(
      matrixFrom([
        [1, 2, 3],
        [10, 20, 30],
      ]),
    );

    for (let i = 0; i < normalized.vectors; i++) {
      const row = normalized.values.slice(
        i * normalized.dimensions,
        (i + 1) * normalized.dimensions,
      );
      const { average, standardDeviation } = calculateStandardDeviation(row);

      expect(average).toBeCloseTo(0, 5);
      expect(standardDeviation).toBeCloseTo(1, 5);
    }
  });

  it("treats each row independently", () => {
    const normalized = normalize(
      matrixFrom([
        [1, 2, 3],
        [101, 102, 103],
        [10, 20, 30],
      ]),
    );

    for (let j = 0; j < normalized.dimensions; j++) {
      const baseVal = normalized.values[getFlatIndex(0, j, normalized.dimensions)]!;
      const shiftedVal = normalized.values[getFlatIndex(1, j, normalized.dimensions)]!;
      const scaledVal = normalized.values[getFlatIndex(2, j, normalized.dimensions)]!;

      expect(baseVal).toBeCloseTo(shiftedVal, 5);
      expect(baseVal).toBeCloseTo(scaledVal, 5);
    }
  });

  it("returns a new matrix without mutating the input", () => {
    const matrix = matrixFrom([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const originalValues = new Float32Array(matrix.values);

    const normalized = normalize(matrix);

    expect(matrix.values).toEqual(originalValues);
    expect(normalized.values).not.toBe(matrix.values);
  });

  it("gracefully handles constant vectors", () => {
    const normalized = normalize(matrixFrom([[3, 3, 3, 3]]));

    for (let j = 0; j < normalized.dimensions; j++) {
      expect(normalized.values[j]).toBeCloseTo(0, 10);
    }
  });
});

describe("addMatrices", () => {
  it("adds matrices element-wise", () => {
    expectMatrixCloseTo(
      addMatrices(
        matrixFrom([
          [1, 2],
          [3, 4],
        ]),
        matrixFrom([
          [10, 20],
          [30, 40],
        ]),
      ),
      matrixFrom([
        [11, 22],
        [33, 44],
      ]),
    );
  });

  it("handles negative and decimal values", () => {
    expectMatrixCloseTo(
      addMatrices(
        matrixFrom([
          [-1.5, 0],
          [2.25, -3],
        ]),
        matrixFrom([
          [0.5, 4],
          [-2.25, 1.5],
        ]),
      ),
      matrixFrom([
        [-1, 4],
        [0, -1.5],
      ]),
    );
  });
});

describe("addVectorsInMatrix", () => {
  it("sums all rows into one vector", () => {
    const result = addVectorsInMatrix(
      matrixFrom([
        [1, 2, 3],
        [4, 5, 6],
        [-2, 10, 0],
      ]),
    );

    expect(result.vectors).toBe(1);
    expect(result.dimensions).toBe(3);
    expect(result.values[0]).toBeCloseTo(3, 10);
    expect(result.values[1]).toBeCloseTo(17, 10);
    expect(result.values[2]).toBeCloseTo(9, 10);
  });
});

describe("concatenateMatricesVertically", () => {
  it("joins matching rows from each same-sized matrix into wider rows", () => {
    expectMatrixCloseTo(
      concatenateMatricesVertically([
        matrixFrom([
          [1, 2],
          [3, 4],
        ]),
        matrixFrom([
          [10, 20],
          [30, 40],
        ]),
        matrixFrom([
          [100, 200],
          [300, 400],
        ]),
      ]),
      matrixFrom([
        [1, 2, 10, 20, 100, 200],
        [3, 4, 30, 40, 300, 400],
      ]),
    );
  });

  it("returns a new matrix without mutating the inputs", () => {
    const left = matrixFrom([
      [1],
      [2],
    ]);
    const right = matrixFrom([
      [3],
      [4],
    ]);
    const leftOriginal = new Float32Array(left.values);
    const rightOriginal = new Float32Array(right.values);

    const concatenated = concatenateMatricesVertically([left, right]);

    expectMatrixCloseTo(concatenated, matrixFrom([
      [1, 3],
      [2, 4],
    ]));
    expect(left.values).toEqual(leftOriginal);
    expect(right.values).toEqual(rightOriginal);
  });
});

describe("sliceRows", () => {
  it("slices the same dimension range from every row", () => {
    expectMatrixCloseTo(
      sliceRows(
        matrixFrom([
          [1, 2, 3, 4],
          [10, 20, 30, 40],
        ]),
        1,
        3,
      ),
      matrixFrom([
        [2, 3],
        [20, 30],
      ]),
    );
  });

  it("returns a new matrix without mutating the input", () => {
    const matrix = matrixFrom([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const originalValues = new Float32Array(matrix.values);

    const sliced = sliceRows(matrix, 0, 2);

    expectMatrixCloseTo(sliced, matrixFrom([
      [1, 2],
      [4, 5],
    ]));
    expect(matrix.values).toEqual(originalValues);
  });
});

describe("sliceToEqualSizes", () => {
  it("splits each row into the requested number of equal-width matrices", () => {
    const sections = sliceToEqualSizes(
      matrixFrom([
        [1, 2, 3, 4, 5, 6],
        [10, 20, 30, 40, 50, 60],
      ]),
      3,
    );

    expect(sections).toHaveLength(3);
    expectMatrixCloseTo(sections[0]!, matrixFrom([
      [1, 2],
      [10, 20],
    ]));
    expectMatrixCloseTo(sections[1]!, matrixFrom([
      [3, 4],
      [30, 40],
    ]));
    expectMatrixCloseTo(sections[2]!, matrixFrom([
      [5, 6],
      [50, 60],
    ]));
  });

  it("handles a single section by returning one full-width matrix", () => {
    const sections = sliceToEqualSizes(
      matrixFrom([
        [1, 2, 3],
        [4, 5, 6],
      ]),
      1,
    );

    expect(sections).toHaveLength(1);
    expectMatrixCloseTo(sections[0]!, matrixFrom([
      [1, 2, 3],
      [4, 5, 6],
    ]));
  });

  it("throws when the row width cannot be split evenly", () => {
    expect(() =>
      sliceToEqualSizes(matrixFrom([[1, 2, 3, 4, 5]]), 2),
    ).toThrow("Can't perfectly divide the nominator 5 by denominator (2)");
  });
});
