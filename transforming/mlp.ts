import { relu } from "../shared/math.ts";
import {
  addVectors,
  multiplyMatrices,
  validateSize,
} from "../shared/matrices.ts";
import type { MultilayerPerceptronWeights } from "../model/model-types.ts";

export const getMultilayerPerceptronUpdateMatrix = (
  encoding: number[][],
  perceptron: MultilayerPerceptronWeights,
  mlpMultiple: number,
) => {
  return encoding.map((vector) =>
    getMultilayerPerceptronUpdateVector(vector, perceptron, mlpMultiple),
  );
};

export const getMultilayerPerceptronUpdateVector = (
  encodingVector: number[],
  perceptron: MultilayerPerceptronWeights,
  mlpMultiple: number,
) => {
  // We identify knowledge from the incoming vectors
  const upped = multiplyMatrices(
    [encodingVector],
    perceptron.wUp.weightsMatrix,
  );

  const hiddenDimensionSize = encodingVector.length;

  validateSize(upped, 1, hiddenDimensionSize * mlpMultiple);

  // We normalize directions
  const uppedBiased = addVectors(upped[0]!, perceptron.wUp.biasVector);

  // We activate neurons
  const nonLinearalized = relu(uppedBiased);

  // We select new knowledge to enrich
  const downed = multiplyMatrices(
    [nonLinearalized],
    perceptron.wDown.weightsMatrix,
  );

  validateSize(downed, 1, hiddenDimensionSize);

  // Not sure what this bias does
  const downedBiased = addVectors(downed[0]!, perceptron.wDown.biasVector);

  return downedBiased;
};
