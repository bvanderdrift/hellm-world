import type {
  AttentionActivations,
  AttentionHeadActivations,
} from "../../model/activations-types.ts";
import type { AttentionWeights } from "../../model/model-types.ts";
import {
  sliceToEqualSizes,
  concatenateMatricesVertically,
  addMatrices,
  applyScalarToVector,
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

export const attentionHeadBackprop = (
  activations: AttentionHeadActivations,
  outputGradients: number[][],
): {
  inputKGradients: number[][];
  inputVGradients: number[][];
  inputQGradients: number[][];
} => {
  const contextLength = activations.output.length;
  const headDimensionality = activations.output[0]?.length!;
  const emptyMatrix = new Array<number[]>(contextLength).fill(
    new Array(headDimensionality).fill(0),
  );

  return outputGradients.reduce(
    (acc, outputGradientVector, vectorIndex) => {
      const valueScalarInputActivations =
        activations.softmaxOutput[vectorIndex]!;

      /**
       * Since `updateVector` is formulized as x_i = a_i + b_i + c_i + ...
       *
       * We can for each a_i etc say
       *
       * dL/da_i = dL/dx_i * dx_i/da_i = dL/dx_i * 1
       *
       * So can just directly take the outputGradientVector for next backprop step
       */

      /**
       * So we just do
       * x_i = w * v_i where w is the weight from the softmax and v_i is the value input activation
       *
       * So if we want to find the input gradient of v_i we do
       *
       * dL/dv_i = dL/dx_i * dx_i/dv_i = dL/dx_i * w
       */
      const inputVGradients = activations.inputV.map((_, valueVectorIndex) => {
        if (valueVectorIndex > vectorIndex) {
          return new Array(headDimensionality).fill(0);
        }

        const scalar = valueScalarInputActivations[valueVectorIndex]!;

        return applyScalarToVector(scalar, outputGradientVector);
      });

      const inputVGradientsCombined = addMatrices(
        // Add one more 0-vector for new index
        acc.inputVGradients,
        inputVGradients,
      );

      return {
        ...acc,
        inputVGradients: inputVGradientsCombined,
      };
    },
    {
      inputKGradients: emptyMatrix,
      inputVGradients: emptyMatrix,
      inputQGradients: emptyMatrix,
    },
  );
};
