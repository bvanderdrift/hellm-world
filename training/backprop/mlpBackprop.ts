import type { MultilayerPerceptronActivations } from "../../model/activations-types.ts";
import type { MultilayerPerceptronWeights } from "../../model/model-types.ts";
import {
  getMatrixSize,
  validateSize,
  addVectorsInMatrix,
} from "../../shared/matrices.ts";
import { matrixBackprop } from "./matrixBackprop.ts";

export const backpropMlp = (
  weights: MultilayerPerceptronWeights,
  activations: MultilayerPerceptronActivations,
  /** C x D matrix */
  outputGradients: number[][],
): {
  inputActivationGradients: number[][];
  weightGradients: MultilayerPerceptronWeights;
} => {
  // dL/db = dL/dMLP * dMLP/db = outputGradients * 1
  const downBiasGradient = addVectorsInMatrix(outputGradients);

  const {
    weightGradients: downWeightsGradients,
    activationGradients: downInputActivationsGradients,
  } = matrixBackprop(
    weights.wDown.weightsMatrix,
    activations.nonLinearToDowning,
    outputGradients,
  );

  const upOutputGradients = reluBackprop(
    activations.uppingToNonLinear,
    downInputActivationsGradients,
  );

  const upBiasGradient = addVectorsInMatrix(upOutputGradients);

  const {
    weightGradients: upWeightsGradients,
    activationGradients: inputActivationGradients,
  } = matrixBackprop(
    weights.wUp.weightsMatrix,
    activations.normalizedInputToUpping,
    upOutputGradients,
  );

  return {
    inputActivationGradients,
    weightGradients: {
      wUp: {
        biasVector: upBiasGradient,
        weightsMatrix: upWeightsGradients,
      },
      wDown: {
        biasVector: downBiasGradient,
        weightsMatrix: downWeightsGradients,
      },
    },
  };
};

export const reluBackprop = (
  inputActivations: number[][],
  outputGradients: number[][],
) => {
  const inputSize = getMatrixSize(inputActivations);
  validateSize(
    outputGradients,
    inputSize.vectorCount,
    inputSize.dimensionsCount,
  );

  return outputGradients.map((gradientVector, vectorIndex) =>
    gradientVector.map((gradient, dimensionIndex) => {
      const inputActivation = inputActivations[vectorIndex]![dimensionIndex]!;

      return inputActivation > 0 ? gradient : 0;
    }),
  );
};
