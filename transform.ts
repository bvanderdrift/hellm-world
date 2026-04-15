import { relu } from "./math.ts";
import { addVectors, multiplyMatrices, validateSize } from "./matrices.ts";
import { type MultilayerPerceptronWeights } from "./weights.ts";

export const runMultilayerPerceptronOnMatrix = (
  encoding: number[][],
  perceptron: MultilayerPerceptronWeights,
) => {
  return encoding.map((vector) =>
    runMultilayerPerceptronOnVector(vector, perceptron),
  );
};

export const runMultilayerPerceptronOnVector = (
  encodingVector: number[],
  perceptron: MultilayerPerceptronWeights,
) => {
  // We identify knowledge from the incoming vectors
  const upped = multiplyMatrices(
    [encodingVector],
    perceptron.wUp.weightsMatrix,
  );

  const hiddenDimensionSize = encodingVector.length;

  validateSize(upped, 1, hiddenDimensionSize * 4);

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

  // We enrich the incoming knowledge with the new knowledge
  return addVectors(encodingVector, downedBiased);
};
