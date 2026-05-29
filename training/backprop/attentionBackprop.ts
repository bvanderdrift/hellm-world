import type {
  AttentionActivations,
  AttentionHeadActivations,
} from "../../model/activations-types.ts";
import type { AttentionWeights } from "../../model/model-types.ts";
import { divideToWhole, dotProduct } from "../../shared/math.ts";
import {
  addMatrices,
  transpose,
  type Matrix,
  createMatrix,
  getFlatIndex,
  getRawVector,
  sliceVectorsFromMatrix,
  sliceRows,
} from "../../shared/matrices.ts";
import { matrixBackprop } from "./matrixBackprop.ts";
import { softmaxBackprop } from "./softmaxBackprop.ts";

export const attentionBackprop = (
  weights: AttentionWeights,
  outputGradients: Matrix,
  activations: AttentionActivations,
  headsCount: number,
): {
  weightGradients: AttentionWeights;
  inputGradients: Matrix;
} => {
  const {
    weightGradients: outMatrixWeightGradients,
    activationGradients: outMatrixInputGradients,
  } = matrixBackprop(
    weights.out,
    activations.outMatrixInputActivations,
    outputGradients,
  );

  const { inputKGradients, inputVGradients, inputQGradients } =
    attentionHeadsBackprop(
      activations.headsActivations,
      outMatrixInputGradients,
      headsCount,
    );

  const {
    activationGradients: kInputGradients,
    weightGradients: kWeightGradients,
  } = matrixBackprop(weights.K, activations.normalizedInput, inputKGradients);

  const {
    activationGradients: vInputGradients,
    weightGradients: vWeightGradients,
  } = matrixBackprop(weights.V, activations.normalizedInput, inputVGradients);

  const {
    activationGradients: qInputGradients,
    weightGradients: qWeightGradients,
  } = matrixBackprop(weights.Q, activations.normalizedInput, inputQGradients);

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

export const attentionHeadsBackprop = (
  activations: AttentionHeadActivations,
  outputGradients: Matrix,
  headsCount: number,
): AttentionHeadInputGradients => {
  let inputKGradients = createMatrix(
    outputGradients.vectors,
    outputGradients.dimensions,
  );
  let inputVGradients = createMatrix(
    outputGradients.vectors,
    outputGradients.dimensions,
  );
  const inputQGradients = createMatrix(
    outputGradients.vectors,
    outputGradients.dimensions,
  );

  const contextLength = outputGradients.vectors;
  const headDimensionality = divideToWhole(
    outputGradients.dimensions,
    headsCount,
  );

  for (
    let vectorIndex = 0;
    vectorIndex < outputGradients.vectors;
    vectorIndex++
  ) {
    const softmaxLength = headsCount * (vectorIndex + 1);

    const softmaxOutputGradients = getSoftmaxOutputGradients(
      outputGradients,
      activations,
      headsCount,
      softmaxLength,
      vectorIndex,
      headDimensionality,
    );
    const softmaxInputGradients = getSoftmaxInputGradients(
      softmaxOutputGradients,
      activations,
      headsCount,
      softmaxLength,
      vectorIndex,
      contextLength,
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

    const { inputKGradients: localInputKGradients, inputQGradientsMatrix } =
      getQKInputGradients(
        outputGradients,
        relevancyVectorGradients,
        activations,
        vectorIndex,
        headsCount,
      );

    inputKGradients = addMatrices(inputKGradients, localInputKGradients);
    inputVGradients = addMatrices(
      inputVGradients,
      getInputVGradients(outputGradients, activations, vectorIndex, headsCount),
    );

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

export const getQKInputGradients = (
  outputGradients: Matrix,
  relevancyVectorGradients: Matrix,
  activations: AttentionHeadActivations,
  vectorIndex: number,
  headsCount: number,
) => {
  const headDimensionality = divideToWhole(
    outputGradients.dimensions,
    headsCount,
  );
  const contextLength = outputGradients.vectors;

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

  const inputKGradientsTransposed = createMatrix(
    outputGradients.dimensions,
    outputGradients.vectors,
  );
  const inputQGradientsMatrix = createMatrix(1, outputGradients.dimensions);

  for (let h = 0; h < headsCount; h++) {
    /**
     * Since the relevancy vectors are just Q_i @ K_0...i we can find gradients of both Q and K inputs directly using the existing matrixBackprop
     */
    const {
      weightGradients: headInputKGradientsTransposed,
      activationGradients: headInputQGradientsMatrix,
    } = matrixBackprop(
      transpose(
        sliceRows(
          lookbackKeys,
          h * headDimensionality,
          (h + 1) * headDimensionality,
        ),
      ),
      sliceRows(
        inputQAtVector,
        h * headDimensionality,
        (h + 1) * headDimensionality,
      ),
      sliceRows(
        relevancyVectorGradients,
        h * (vectorIndex + 1),
        (h + 1) * (vectorIndex + 1),
      ),
    );

    // We can't use direct flat `set` b/c `headInputKGradientsTransposed` is width 'vectorIndex + 1' and `inputKGradientsTransposed` is width 'contextLength'
    for (let m = 0; m < headDimensionality; m++) {
      inputKGradientsTransposed.values.set(
        headInputKGradientsTransposed.values.slice(
          m * (vectorIndex + 1),
          (m + 1) * (vectorIndex + 1),
        ),
        getFlatIndex(h * headDimensionality + m, 0, contextLength),
      );
    }

    inputQGradientsMatrix.values.set(
      headInputQGradientsMatrix.values,
      h * headDimensionality,
    );
  }

  if (inputQGradientsMatrix.vectors !== 1) {
    throw new Error(`Expected only a single vector from Q vector backprop`);
  }

  const inputKGradients = transpose(inputKGradientsTransposed);

  return { inputKGradients, inputQGradientsMatrix };
};

export const getSoftmaxOutputGradients = (
  outputGradients: Matrix,
  activations: AttentionHeadActivations,
  headsCount: number,
  softmaxLength: number,
  vectorIndex: number,
  headDimensionality: number,
) => {
  const softmaxOutputGradients = new Float32Array(softmaxLength);

  for (
    let lookbackTokenIndex = 0;
    lookbackTokenIndex < vectorIndex + 1;
    lookbackTokenIndex++
  ) {
    for (let h = 0; h < headsCount; h++) {
      const offsetDimensionality = h * headDimensionality;
      const offsetTokens = h * (vectorIndex + 1);

      let dotProductValue = 0;

      for (let j = 0; j < headDimensionality; j++) {
        dotProductValue +=
          outputGradients.values[
            getFlatIndex(
              vectorIndex,
              offsetDimensionality + j,
              outputGradients.dimensions,
            )
          ]! *
          activations.inputV.values[
            getFlatIndex(
              lookbackTokenIndex,
              offsetDimensionality + j,
              activations.inputV.dimensions,
            )
          ]!;
      }

      softmaxOutputGradients[offsetTokens + lookbackTokenIndex] =
        dotProductValue;
    }
  }

  return softmaxOutputGradients;
};

export const getSoftmaxInputGradients = (
  softmaxOutputGradients: Float32Array,
  activations: AttentionHeadActivations,
  headsCount: number,
  softmaxLength: number,
  vectorIndex: number,
  contextLength: number,
) => {
  const softmaxInputGradients = new Float32Array(softmaxLength);

  const softmaxInput = getRawVector(
    activations.attentionRelevancyOutput,
    vectorIndex,
  );

  for (let h = 0; h < headsCount; h++) {
    const offset = h * contextLength;

    const headSoftmaxInput = softmaxInput.slice(
      offset,
      offset + vectorIndex + 1,
    );
    const headSoftmaxOuputGradient = softmaxOutputGradients.slice(
      h * (vectorIndex + 1),
      (h + 1) * (vectorIndex + 1),
    );

    const headSoftmaxInputGradient = softmaxBackprop(
      headSoftmaxInput,
      headSoftmaxOuputGradient,
    );

    softmaxInputGradients.set(headSoftmaxInputGradient, h * (vectorIndex + 1));
  }

  return softmaxInputGradients;
};

export const getInputVGradients = (
  outputGradients: Matrix,
  activations: AttentionHeadActivations,
  vectorIndex: number,
  headsCount: number,
) => {
  const contextLength = outputGradients.vectors;
  const headDimensionality = divideToWhole(
    outputGradients.dimensions,
    headsCount,
  );

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
    for (let h = 0; h < headsCount; h++) {
      const contextOffset = h * contextLength;

      const scalar =
        activations.softmaxOutput.values[
          getFlatIndex(
            vectorIndex,
            contextOffset + valueVectorIndex,
            activations.softmaxOutput.dimensions,
          )
        ]!;

      const offset = h * headDimensionality;

      for (let j = 0; j < headDimensionality; j++) {
        localInputVGradients.values[
          getFlatIndex(
            valueVectorIndex,
            offset + j,
            localInputVGradients.dimensions,
          )
        ] =
          scalar *
          outputGradients.values[
            getFlatIndex(vectorIndex, offset + j, outputGradients.dimensions)
          ]!;
      }
    }
  }

  return localInputVGradients;
};
