import { operateOnMatrices, operateOnMatrix, type Matrix } from "./matrices.ts";

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

export const safeSumExponatedLogits = (logits: Float32Array) => {
  const biggestLogit = Math.max(...logits);
  // To prevent overflows. Logits are still related the same since they all subtract the same value
  const safeLogits = logits.map((logit) => logit - biggestLogit);
  const exponatedLogits = safeLogits.map((l) => Math.exp(l));

  return {
    safeLogits,
    exponatedLogits,
    summed: sum(exponatedLogits),
    biggestLogit,
  };
};

/**
 * s_i = e^l_i / sum(e^l_j)
 */
export const softmax = (logits: Float32Array) => {
  const { safeLogits, summed } = safeSumExponatedLogits(logits);

  return safeLogits.map((logit) => Math.exp(logit) / summed);
};

/** Rectified Linear Unit */
export const relu = (matrix: Matrix) =>
  operateOnMatrix(matrix, (value) => Math.max(value, 0));

export const mean = (values: Float32Array) => {
  return sum(values) / values.length;
};

export const calculateStandardDeviation = (values: Float32Array) => {
  const average = mean(values);

  const squareDeltas = values.map((value) => Math.pow(value - average, 2));

  const averageSquareDeltas = mean(squareDeltas);

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
