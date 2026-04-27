import { relu } from "../shared/math.ts";
import {
  addVectors,
  getMatrixSize,
  multiplyMatrices,
  validateSize,
} from "../shared/matrices.ts";
import type { MultilayerPerceptronWeights } from "../model/model-types.ts";

export const getMultilayerPerceptronUpdateMatrix = (
  encoding: number[][],
  perceptron: MultilayerPerceptronWeights,
  mlpMultiple: number,
) => {
  const inputSize = getMatrixSize(encoding);

  const upped = multiplyMatrices(encoding, perceptron.wUp.weightsMatrix);

  validateSize(
    upped,
    inputSize.vectorCount,
    inputSize.dimensionsCount * mlpMultiple,
  );

  // We normalize directions
  const uppedBiased = upped.map((upVector) =>
    addVectors(upVector, perceptron.wUp.biasVector),
  );

  // We activate neurons
  const nonLinearalized = relu(uppedBiased);

  // We select new knowledge to enrich
  const downed = multiplyMatrices(
    nonLinearalized,
    perceptron.wDown.weightsMatrix,
  );

  validateSize(downed, inputSize.vectorCount, inputSize.dimensionsCount);

  // Not sure what this bias does
  const downedBiased = downed.map((downedVector) =>
    addVectors(downedVector, perceptron.wDown.biasVector),
  );

  return downedBiased;
};
