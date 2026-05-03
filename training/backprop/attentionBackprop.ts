import type {
  AttentionActivations,
  AttentionHeadActivations,
} from "../../model/activations-types.ts";
import type { AttentionWeights } from "../../model/model-types.ts";
import { dotProduct } from "../../shared/math.ts";
import {
  sliceToEqualSizes,
  concatenateMatricesVertically,
  addMatrices,
  applyScalarToVector,
  transpose,
} from "../../shared/matrices.ts";
import { matrixBackprop } from "./matrixBackprop.ts";
import { softmaxBackprop } from "./softmaxBackprop.ts";

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

type AttentionHeadInputGradients = {
  inputKGradients: number[][];
  inputVGradients: number[][];
  inputQGradients: number[][];
};

export const attentionHeadBackprop = (
  activations: AttentionHeadActivations,
  outputGradients: number[][],
): AttentionHeadInputGradients => {
  const contextLength = activations.output.length;
  const headDimensionality = activations.output[0]?.length!;
  const emptyMatrix = new Array<number[]>(contextLength).fill(
    new Array(headDimensionality).fill(0),
  );

  return outputGradients.reduce<AttentionHeadInputGradients>(
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
       * dL/dw = sum_j(dL/dx_j * dx_j/dw) = sum_j(dL/dx_j * v_j)
       */
      const inputVGradients = activations.inputV.map((_, valueVectorIndex) => {
        if (valueVectorIndex > vectorIndex) {
          return new Array(headDimensionality).fill(0);
        }

        const scalar = valueScalarInputActivations[valueVectorIndex]!;

        return applyScalarToVector(scalar, outputGradientVector);
      });

      const softmaxOutputGradients = activations.inputV
        .slice(0, vectorIndex + 1)
        .map((inputVAtToken) => {
          return dotProduct(outputGradientVector, inputVAtToken);
        });

      const softmaxInput = activations.attentionRelevancyOutput[vectorIndex]!;
      const softmaxInputGradients = softmaxBackprop(
        softmaxInput,
        softmaxOutputGradients,
      );

      /**
       * Since we do y_i = x_i / Math.sqrt(j)
       * we determine dL/dx_i as
       *
       * dL/dx_i = dL/dy_i * dy_i/dx_i
       *   = softmaxInputGradients_i * 1 / Math.sqrt(j)
       */
      const relevancyVectorGradients = softmaxInputGradients.map(
        (outputGradient) => outputGradient / Math.sqrt(headDimensionality),
      );

      const lookbackKeys = activations.inputK.slice(0, vectorIndex + 1); // inclusive

      /**
       * Since the relevancy vectors are just Q_i @ K_0...i we can find gradients of both Q and K inputs directly using the existing matrixBackprop
       */
      const {
        weightGradients: inputKGradientsTransposed,
        activationGradients: inputQGradientsMatrix,
      } = matrixBackprop(
        transpose(lookbackKeys),
        [activations.inputQ[vectorIndex]!],
        [relevancyVectorGradients],
      );

      if (inputQGradientsMatrix.length !== 1) {
        throw new Error(`Expected only a single vector from Q vector backprop`);
      }

      const inputQGradients = inputQGradientsMatrix[0]!;

      const inputKGradients = [
        ...transpose(inputKGradientsTransposed),
        ...new Array(outputGradients.length - vectorIndex - 1).fill(
          new Array(headDimensionality).fill(0),
        ),
      ];

      return {
        inputKGradients: addMatrices(acc.inputKGradients, inputKGradients),
        inputVGradients: addMatrices(acc.inputVGradients, inputVGradients),
        inputQGradients: [...acc.inputQGradients, inputQGradients],
      };
    },
    {
      inputKGradients: emptyMatrix, // We sum this so create a full starting matrix with 0-values
      inputVGradients: emptyMatrix, // We sum this so create a full starting matrix with 0-values
      inputQGradients: [] satisfies number[][], // We concatenate this so make a 0-vector matrix (empty array)
    },
  );
};
