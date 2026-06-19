import { operateOnMatrix, type Matrix } from "./matrices/matrices.ts";

export const sum = (values: Float32Array) => {
  let output = 0;

  for (let index = 0; index < values.length; index++) {
    output += values[index]!;
  }

  return output;
};

export const dotProduct = (v1: Float32Array, v2: Float32Array) => {
  if (v1.length !== v2.length) {
    throw new Error(
      `v1 lenght ${v1.length} and v2 length ${v2.length} not overlapping`,
    );
  }

  const multiples = v1.map((e, i) => e * v2[i]!);

  return sum(multiples);
};

/** Rectified Linear Unit */
export const relu = (matrix: Matrix) =>
  operateOnMatrix(matrix, (value) => Math.max(value, 0));

export const mean = (values: Float32Array) => {
  return sum(values) / values.length;
};

export const calculateStandardDeviation = (values: Float32Array) => {
  const average = mean(values);

  let summedSquareDeltas = 0;

  for (let index = 0; index < values.length; index++) {
    const value = values[index]!;
    summedSquareDeltas += Math.pow(value - average, 2);
  }

  const averageSquareDeltas = summedSquareDeltas / values.length;

  return {
    average,
    standardDeviation: Math.sqrt(averageSquareDeltas),
  };
};

export const divideToWhole = (nominator: number, denominator: number) => {
  const divisionRemainder = nominator % denominator;

  if (divisionRemainder !== 0) {
    throw new Error(
      `Can't perfectly divide the nominator ${nominator} by denominator (${denominator})`,
    );
  }

  return Math.round(nominator / denominator);
};

export const randomNormalDistribution = (
  mean: number,
  standardDeviation: number,
) => {
  const uniform1 = Math.random() || Number.EPSILON; // Epsilon to prevent a 0 since we're going to take a log
  const uniform2 = Math.random();

  const radius = Math.sqrt(-2 * Math.log(uniform1)); //
  const angle = 2 * Math.PI * uniform2; // Random angle on a circle

  const noMeanStdOf1 = radius * Math.cos(angle);

  return mean + standardDeviation * noMeanStdOf1;
};
