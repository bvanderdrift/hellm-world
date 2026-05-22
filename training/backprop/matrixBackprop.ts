import {
  transpose,
  multiplyMatrices,
  type Matrix,
} from "../../shared/matrices.ts";

export const matrixBackprop = (
  weights: Matrix,
  inputActivations: Matrix,
  outputGradients: Matrix,
) => {
  const inputsByDimension = transpose(inputActivations);

  const weightGradients = multiplyMatrices(inputsByDimension, outputGradients);

  // dL/da = dl/dMLP * dMLP/da = outputGradients * wDownT.
  // Simply because of a function `a * x + b`, if we take the derivative for `a` we simply have `x` which in this case is `w_ij` so we just transpose the weight matrix
  const activationGradients = multiplyMatrices(
    outputGradients,
    transpose(weights),
  );

  return {
    weightGradients,
    activationGradients,
  };
};
