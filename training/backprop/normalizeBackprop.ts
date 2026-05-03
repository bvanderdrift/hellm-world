import { calculateStandardDeviation, sum } from "../../shared/math.ts";

/**
 * Normalize function is f(h_vi) where h_vi is input activation of vector v at index i
 *
 * f(h_vi) = (h_vi - mean(h_v0...j)) / std(h_v0...j)
 * = (1 / std(h_v0...j)) * (h_vi) - (1 / std(h_v0...j)) * mean(h_v0...j)
 * = std^-1 * h_vi - std^-1 * mean
 *
 * So if k is the input index we are taking the derivative for:
 *
 * df/dh_k = d(std^-1 * h_vi)/dh_k - d(std^-1 * mean)/dh_k
 *
 * d(std^-1 * h_vi)/dh_k = d(std^-1)/dh_k * h_vi + std^-1 * dh_vi/dh_k
 * where dh_vi/dh_k = 1 if i === k, otherwise 0
 *
 * and
 *
 * d(std^-1 * mean)/dh_k = d(std^-1)/dh_k * mean + std^-1 * dmean/dh_k
 *  = d(std^-1)/dh_k * mean + std^-1 * 1/j
 *  = d(std^-1)/dh_k * mean + std^-1/j
 *  = d(std^-1)/dh_k * mean + 1 / (j * std)
 *
 * and
 *
 * d(std^-1)/dh_k = d(std^-1)/dstd * dstd/dh_k
 *  = -std^-2 * dstd/dh_k
 *
 * The full backprop step needs every y_i that depends on h_k:
 *
 * dL/dh_k = sum_i(dL/dy_i * dy_i/dh_k)
 */
export const backpropNormalize = (
  outputGradients: number[][],
  inputActivations: number[][],
): number[][] => {
  return inputActivations.map((vector, vectorIndex) =>
    vector.map((_, valueIndex) => {
      const vectorOutputGradients = outputGradients[vectorIndex]!;

      const { average, standardDeviation } = calculateStandardDeviation(vector);

      // To prevent divide-by-0
      const safeStandardDeviation = standardDeviation + Number.EPSILON;

      const vectorInputGradients = vectorOutputGradients.map(
        (outputGradient, gradientIndex) => {
          const dStdDh =
            -(safeStandardDeviation ** -2) *
            standardDeviationDerivative(vector, valueIndex);

          const dhiDhk = valueIndex === gradientIndex ? 1 : 0;

          const hi = vector[gradientIndex]!;

          const firstBranch = dStdDh * hi + dhiDhk / safeStandardDeviation;

          const secondBranch =
            dStdDh * average + 1 / (vector.length * safeStandardDeviation);

          const dyiDh = firstBranch - secondBranch;

          return outputGradient * dyiDh;
        },
      );

      return sum(vectorInputGradients);
    }),
  );
};

/**
 * std(h_v0...j) = Math.sqrt(mean(Math.pow(h_vj - mean(h_v0...j)))
 * = Math.sqrt(variance(h_v0...j))
 * and variance(h_v0...j) = 1/j * sum(Math.pow(h_vj - mean(h_v0...j)))
 * dstd(h_v0...j)/dh_vi = d/dvariance * dvariance/h_vj
 *
 * d/dvariance = (1/ (2 * std))
 *
 * dvariance/dh_vi = 1/j * 2 * (h_vi - mean(h_v0...j))
 *
 * So dstd/dh_i = (1/ (2 * std)) * 2/j * (h_vi - mean(h_v0...j))
 *  = 1 / (j * std) * (h_vi - mean)
 *  = (h_vh - mean) / (j * std)
 */
const standardDeviationDerivative = (
  values: number[],
  derivativeIndex: number,
) => {
  const { average, standardDeviation } = calculateStandardDeviation(values);

  const derivative = values[derivativeIndex]!;

  return (
    (derivative - average) /
    (values.length *
      (standardDeviation +
        // To stabalize againt 0-division
        Number.EPSILON))
  );
};
