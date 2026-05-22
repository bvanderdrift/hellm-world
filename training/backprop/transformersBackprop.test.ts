import type { TransformerActivations } from "../../model/activations-types.ts";
import type { TransformerWeights } from "../../model/model-types.ts";
import { addMatrices } from "../../shared/matrices.ts";
import { matrixFrom } from "../../testing/testing-utils.ts";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  attentionBackprop: vi.fn(),
  backpropMlp: vi.fn(),
  backpropNormalize: vi.fn(),
}));

vi.mock("./attentionBackprop.ts", () => ({
  attentionBackprop: mocks.attentionBackprop,
}));

vi.mock("./mlpBackprop.ts", () => ({
  backpropMlp: mocks.backpropMlp,
}));

vi.mock("./normalizeBackprop.ts", () => ({
  backpropNormalize: mocks.backpropNormalize,
}));

import { transformersBackprop } from "./transformersBackprop.ts";

const makeAttentionWeights = (
  seed: number,
): TransformerWeights["attention"] => {
  return {
    Q: matrixFrom([[seed]]),
    K: matrixFrom([[seed + 1]]),
    V: matrixFrom([[seed + 2]]),
    out: matrixFrom([[seed + 3]]),
  };
};

const makeMlpWeights = (
  seed: number,
): TransformerWeights["multilayerPerceptron"] => {
  return {
    wUp: {
      weightsMatrix: matrixFrom([[seed]]),
      biasVector: matrixFrom([[seed + 1]]),
    },
    wDown: {
      weightsMatrix: matrixFrom([[seed + 2]]),
      biasVector: matrixFrom([[seed + 3]]),
    },
  };
};

const makeTransformerWeights = (seed: number): TransformerWeights => {
  return {
    attention: makeAttentionWeights(seed),
    multilayerPerceptron: makeMlpWeights(seed + 10),
  };
};

const makeTransformerActivations = (seed: number): TransformerActivations => {
  return {
    transformerInput: matrixFrom([
      [seed, seed + 1],
      [seed + 2, seed + 3],
    ]),
    attention: {
      normalizedInput: matrixFrom([
        [seed + 10, seed + 11],
        [seed + 12, seed + 13],
      ]),
      heads: [],
      outMatrixInputActivations: matrixFrom([
        [seed + 20, seed + 21],
        [seed + 22, seed + 23],
      ]),
      output: matrixFrom([
        [seed + 30, seed + 31],
        [seed + 32, seed + 33],
      ]),
    },
    mlp: {
      normalizedInputToUpping: matrixFrom([
        [seed + 40, seed + 41],
        [seed + 42, seed + 43],
      ]),
      uppingToNonLinear: matrixFrom([
        [seed + 50, seed + 51],
        [seed + 52, seed + 53],
      ]),
      nonLinearToDowning: matrixFrom([
        [seed + 60, seed + 61],
        [seed + 62, seed + 63],
      ]),
      downingOutput: matrixFrom([
        [seed + 70, seed + 71],
        [seed + 72, seed + 73],
      ]),
    },
  };
};

describe("transformersBackprop", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("adds residual gradients around both transformer branches", () => {
    const outputGradients = matrixFrom([
      [10, 20],
      [30, 40],
    ]);
    const weights = [makeTransformerWeights(1)];
    const activations = [makeTransformerActivations(100)];
    const mlpInputGradients = matrixFrom([
      [1, 2],
      [3, 4],
    ]);
    const mlpNormInputGradients = matrixFrom([
      [5, 6],
      [7, 8],
    ]);
    const attentionInputGradients = matrixFrom([
      [9, 10],
      [11, 12],
    ]);
    const attentionNormInputGradients = matrixFrom([
      [13, 14],
      [15, 16],
    ]);
    const attentionOutputGradients = addMatrices(
      outputGradients,
      mlpNormInputGradients,
    );

    mocks.backpropMlp.mockReturnValue({
      inputActivationGradients: mlpInputGradients,
      weightGradients: makeMlpWeights(200),
    });
    mocks.backpropNormalize
      .mockReturnValueOnce(mlpNormInputGradients)
      .mockReturnValueOnce(attentionNormInputGradients);
    mocks.attentionBackprop.mockReturnValue({
      inputGradients: attentionInputGradients,
      weightGradients: makeAttentionWeights(300),
    });

    const gradients = transformersBackprop(
      outputGradients,
      weights,
      activations,
    );

    expect(mocks.attentionBackprop).toHaveBeenCalledWith(
      weights[0]!.attention,
      attentionOutputGradients,
      activations[0]!.attention,
    );
    expect(mocks.backpropNormalize.mock.calls[0]![0]).toEqual(
      mlpInputGradients,
    );
    expect(mocks.backpropNormalize.mock.calls[1]![0]).toEqual(
      attentionInputGradients,
    );
    expect(mocks.backpropNormalize.mock.calls[1]![1]).toEqual(
      activations[0]!.transformerInput,
    );
    expect(gradients.inputActivationGradients).toEqual(
      addMatrices(attentionOutputGradients, attentionNormInputGradients),
    );
  });

  it("walks layers backward but returns weight gradients in forward layer order", () => {
    const outputGradients = matrixFrom([[1, 2]]);
    const weights = [makeTransformerWeights(1), makeTransformerWeights(100)];
    const activations = [
      makeTransformerActivations(10),
      makeTransformerActivations(200),
    ];
    const layerOneMlpInputGradients = matrixFrom([[3, 4]]);
    const layerOneMlpNormGradients = matrixFrom([[5, 6]]);
    const layerOneAttentionInputGradients = matrixFrom([[7, 8]]);
    const layerOneAttentionNormGradients = matrixFrom([[9, 10]]);
    const layerZeroOutputGradients = addMatrices(
      addMatrices(outputGradients, layerOneMlpNormGradients),
      layerOneAttentionNormGradients,
    );
    const layerZeroMlpInputGradients = matrixFrom([[11, 12]]);
    const layerZeroMlpNormGradients = matrixFrom([[13, 14]]);
    const layerZeroAttentionInputGradients = matrixFrom([[15, 16]]);
    const layerZeroAttentionNormGradients = matrixFrom([[17, 18]]);
    const layerZeroAttentionOutputGradients = addMatrices(
      layerZeroOutputGradients,
      layerZeroMlpNormGradients,
    );
    const layerZeroMlpWeightGradients = makeMlpWeights(500);
    const layerOneMlpWeightGradients = makeMlpWeights(600);
    const layerZeroAttentionWeightGradients = makeAttentionWeights(700);
    const layerOneAttentionWeightGradients = makeAttentionWeights(800);

    mocks.backpropMlp
      .mockReturnValueOnce({
        inputActivationGradients: layerOneMlpInputGradients,
        weightGradients: layerOneMlpWeightGradients,
      })
      .mockReturnValueOnce({
        inputActivationGradients: layerZeroMlpInputGradients,
        weightGradients: layerZeroMlpWeightGradients,
      });
    mocks.backpropNormalize
      .mockReturnValueOnce(layerOneMlpNormGradients)
      .mockReturnValueOnce(layerOneAttentionNormGradients)
      .mockReturnValueOnce(layerZeroMlpNormGradients)
      .mockReturnValueOnce(layerZeroAttentionNormGradients);
    mocks.attentionBackprop
      .mockReturnValueOnce({
        inputGradients: layerOneAttentionInputGradients,
        weightGradients: layerOneAttentionWeightGradients,
      })
      .mockReturnValueOnce({
        inputGradients: layerZeroAttentionInputGradients,
        weightGradients: layerZeroAttentionWeightGradients,
      });

    const gradients = transformersBackprop(
      outputGradients,
      weights,
      activations,
    );

    expect(mocks.backpropMlp).toHaveBeenNthCalledWith(
      1,
      weights[1]!.multilayerPerceptron,
      activations[1]!.mlp,
      outputGradients,
    );
    expect(mocks.attentionBackprop).toHaveBeenNthCalledWith(
      1,
      weights[1]!.attention,
      addMatrices(outputGradients, layerOneMlpNormGradients),
      activations[1]!.attention,
    );
    expect(mocks.backpropMlp).toHaveBeenNthCalledWith(
      2,
      weights[0]!.multilayerPerceptron,
      activations[0]!.mlp,
      layerZeroOutputGradients,
    );
    expect(mocks.attentionBackprop).toHaveBeenNthCalledWith(
      2,
      weights[0]!.attention,
      layerZeroAttentionOutputGradients,
      activations[0]!.attention,
    );
    expect(gradients.transformerGradients).toEqual([
      {
        attention: layerZeroAttentionWeightGradients,
        multilayerPerceptron: layerZeroMlpWeightGradients,
      },
      {
        attention: layerOneAttentionWeightGradients,
        multilayerPerceptron: layerOneMlpWeightGradients,
      },
    ]);
    expect(gradients.inputActivationGradients).toEqual(
      addMatrices(
        layerZeroAttentionOutputGradients,
        layerZeroAttentionNormGradients,
      ),
    );
  });

  it("uses the pre-MLP residual state when backpropagating the MLP layer norm", () => {
    const outputGradients = matrixFrom([[1, 2]]);
    const weights = [makeTransformerWeights(1)];
    const activations = [makeTransformerActivations(100)];
    const expectedMlpNormalizationInput = addMatrices(
      activations[0]!.transformerInput,
      activations[0]!.attention.output,
    );

    mocks.backpropMlp.mockReturnValue({
      inputActivationGradients: matrixFrom([[3, 4]]),
      weightGradients: makeMlpWeights(200),
    });
    mocks.backpropNormalize
      .mockReturnValueOnce(matrixFrom([[5, 6]]))
      .mockReturnValueOnce(matrixFrom([[7, 8]]));
    mocks.attentionBackprop.mockReturnValue({
      inputGradients: matrixFrom([[9, 10]]),
      weightGradients: makeAttentionWeights(300),
    });

    transformersBackprop(outputGradients, weights, activations);

    expect(mocks.backpropNormalize.mock.calls[0]![1]).toEqual(
      expectedMlpNormalizationInput,
    );
  });
});
