import type {
  AttentionActivations,
  AttentionHeadActivations,
} from "../../model/activations-types.ts";
import type { AttentionWeights } from "../../model/model-types.ts";
import {
  sliceToEqualSizes,
  concatenateMatricesVertically,
  addMatrices,
} from "../../shared/matrices.ts";
import { matrixBackprop } from "./matrixBackprop.ts";

export const attentionBackprop = (
  weights: AttentionWeights,
  outputGradients: number[][],
  activations: AttentionActivations,
): {
  weightGradients: AttentionWeights;
  inputGradients: number[][];
} => {
  const headsCount = activations.heads.length;

  const {
    weightGradients: outMatrixWeightGradients,
    activationGradients: outMatrixInputGradients,
  } = matrixBackprop(
    weights.out,
    activations.outMatrixInputActivations,
    outputGradients,
  );

  const headOutputMatrixes = sliceToEqualSizes(
    outMatrixInputGradients,
    headsCount,
  );

  const headsInputGradients = activations.heads.map(
    (headActivations, headIndex) =>
      attentionHeadBackprop(headActivations, headOutputMatrixes[headIndex]!),
  );

  const kOutputGradients = concatenateMatricesVertically(
    headsInputGradients.map((h) => h.inputKGradients),
  );
  const {
    activationGradients: kInputGradients,
    weightGradients: kWeightGradients,
  } = matrixBackprop(weights.K, activations.normalizedInput, kOutputGradients);

  const vOutputGradients = concatenateMatricesVertically(
    headsInputGradients.map((h) => h.inputVGradients),
  );
  const {
    activationGradients: vInputGradients,
    weightGradients: vWeightGradients,
  } = matrixBackprop(weights.V, activations.normalizedInput, vOutputGradients);

  const qOutputGradients = concatenateMatricesVertically(
    headsInputGradients.map((h) => h.inputQGradients),
  );
  const {
    activationGradients: qInputGradients,
    weightGradients: qWeightGradients,
  } = matrixBackprop(weights.Q, activations.normalizedInput, qOutputGradients);

  return {
    weightGradients: {
      out: outMatrixWeightGradients,
      K: kWeightGradients,
      V: vWeightGradients,
      Q: qWeightGradients,
    },
    inputGradients: addMatrices(
      addMatrices(kInputGradients, vInputGradients),
      qInputGradients,
    ),
  };
};

const attentionHeadBackprop = (
  activations: AttentionHeadActivations,
  outputGradients: number[][],
): {
  inputKGradients: number[][];
  inputVGradients: number[][];
  inputQGradients: number[][];
} => {};
