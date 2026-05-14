import { relu } from "../shared/math.ts";
import {
  addVectorAcrossMatrix,
  multiplyMatrices,
  type Matrix,
} from "../shared/matrices.ts";
import type { MultilayerPerceptronWeights } from "../model/model-types.ts";
import type { MultilayerPerceptronActivations } from "../model/activations-types.ts";

export const getMultilayerPerceptronActivations = (
  encoding: Matrix,
  perceptron: MultilayerPerceptronWeights,
): MultilayerPerceptronActivations => {
  const upped = multiplyMatrices(encoding, perceptron.wUp.weightsMatrix);

  // We normalize directions
  const uppedBiased = addVectorAcrossMatrix(perceptron.wUp.biasVector, upped);

  // We activate neurons
  const nonLinearalized = relu(uppedBiased);

  // We select new knowledge to enrich
  const downed = multiplyMatrices(
    nonLinearalized,
    perceptron.wDown.weightsMatrix,
  );

  // Not sure what this bias does
  const downedBiased = addVectorAcrossMatrix(
    perceptron.wDown.biasVector,
    downed,
  );

  return {
    normalizedInputToUpping: encoding,
    uppingToNonLinear: uppedBiased,
    nonLinearToDowning: nonLinearalized,
    downingOutput: downedBiased,
  };
};
