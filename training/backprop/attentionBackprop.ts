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
  transpose,
  type Matrix,
  createMatrix,
  getFlatIndex,
  getRawVector,
  sliceVectorsFromMatrix,
} from "../../shared/matrices.ts";
import { matrixBackprop } from "./matrixBackprop.ts";
import { softmaxBackprop } from "./softmaxBackprop.ts";

export const attentionBackprop = (
  weights: AttentionWeights,
  outputGradients: Matrix,
  activations: AttentionActivations,
): {
  weightGradients: AttentionWeights;
  inputGradients: Matrix;
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
  inputKGradients: Matrix;
  inputVGradients: Matrix;
  inputQGradients: Matrix;
};

export const attentionHeadBackprop = (
  activations: AttentionHeadActivations,
  outputGradients: Matrix,
): AttentionHeadInputGradients => {
  let inputKGradients = createMatrix(
    outputGradients.vectors,
    outputGradients.dimensions,
  );
  let inputVGradients = createMatrix(
    outputGradients.vectors,
    outputGradients.dimensions,
  );
  let inputQGradients = createMatrix(
    outputGradients.vectors,
    outputGradients.dimensions,
  );

  const contextLength = outputGradients.vectors;
  const headDimensionality = outputGradients.dimensions;

  for (
    let vectorIndex = 0;
    vectorIndex < outputGradients.vectors;
    vectorIndex++
  ) {
    const localInputVGradients = createMatrix(
      outputGradients.vectors,
      outputGradients.dimensions,
    );

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
    for (
      let valueVectorIndex = 0;
      valueVectorIndex < vectorIndex + 1;
      valueVectorIndex++
    ) {
      const scalar =
        activations.softmaxOutput.values[
          getFlatIndex(vectorIndex, valueVectorIndex, contextLength)
        ]!;

      for (let j = 0; j < headDimensionality; j++) {
        localInputVGradients.values[
          getFlatIndex(valueVectorIndex, j, headDimensionality)
        ] =
          scalar *
          outputGradients.values[
            getFlatIndex(vectorIndex, j, headDimensionality)
          ]!;
      }
    }

    const softmaxOutputGradients = new Float32Array(vectorIndex + 1);

    for (
      let lookbackTokenIndex = 0;
      lookbackTokenIndex < vectorIndex + 1;
      lookbackTokenIndex++
    ) {
      softmaxOutputGradients[lookbackTokenIndex] = dotProduct(
        getRawVector(outputGradients, vectorIndex),
        getRawVector(activations.inputV, lookbackTokenIndex),
      );
    }

    const softmaxInput = getRawVector(
      activations.attentionRelevancyOutput,
      vectorIndex,
    ).slice(0, vectorIndex + 1);
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
    const relevancyVectorGradients = createMatrix(
      1,
      softmaxInputGradients.length,
    );

    for (let index = 0; index < relevancyVectorGradients.dimensions; index++) {
      relevancyVectorGradients.values[index] =
        softmaxInputGradients[index]! / Math.sqrt(headDimensionality);
    }

    const lookbackKeys = sliceVectorsFromMatrix(
      activations.inputK,
      0,
      vectorIndex + 1,
    ); // inclusive
    const inputQAtVector = sliceVectorsFromMatrix(
      activations.inputQ,
      vectorIndex,
      vectorIndex + 1,
    );

    /**
     * Since the relevancy vectors are just Q_i @ K_0...i we can find gradients of both Q and K inputs directly using the existing matrixBackprop
     */
    const {
      weightGradients: inputKGradientsTransposed,
      activationGradients: inputQGradientsMatrix,
    } = matrixBackprop(
      transpose(lookbackKeys),
      inputQAtVector,
      relevancyVectorGradients,
    );

    if (inputQGradientsMatrix.vectors !== 1) {
      throw new Error(`Expected only a single vector from Q vector backprop`);
    }

    const localInputKGradients = createMatrix(
      outputGradients.vectors,
      outputGradients.dimensions,
    );
    const inputKGradientsPartial = transpose(inputKGradientsTransposed);

    for (let i = 0; i < inputKGradientsPartial.values.length; i++) {
      localInputKGradients.values[i] = inputKGradientsPartial.values[i]!;
    }

    inputKGradients = addMatrices(inputKGradients, localInputKGradients);
    inputVGradients = addMatrices(inputVGradients, localInputVGradients);

    for (let j = 0; j < outputGradients.dimensions; j++) {
      inputQGradients.values[
        getFlatIndex(vectorIndex, j, outputGradients.dimensions)
      ] = inputQGradientsMatrix.values[j]!;
    }
  }

  return {
    inputKGradients,
    inputVGradients,
    inputQGradients,
  };
};
