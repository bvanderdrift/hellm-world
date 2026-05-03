import type { TransformerActivations } from "../../model/activations-types.ts";
import type { TransformerWeights } from "../../model/model-types.ts";
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

const addMatrices = (left: number[][], right: number[][]) =>
  left.map((row, rowIndex) =>
    row.map((value, dimensionIndex) => {
      return value + right[rowIndex]![dimensionIndex]!;
    }),
  );

const makeAttentionWeights = (seed: number): TransformerWeights["attention"] => {
  return {
    Q: [[seed]],
    K: [[seed + 1]],
    V: [[seed + 2]],
    out: [[seed + 3]],
  };
};

const makeMlpWeights = (
  seed: number,
): TransformerWeights["multilayerPerceptron"] => {
  return {
    wUp: {
      weightsMatrix: [[seed]],
      biasVector: [seed + 1],
    },
    wDown: {
      weightsMatrix: [[seed + 2]],
      biasVector: [seed + 3],
    },
  };
};

const makeTransformerWeights = (seed: number): TransformerWeights => {
  return {
    attention: makeAttentionWeights(seed),
    multilayerPerceptron: makeMlpWeights(seed + 10),
  };
};

const makeTransformerActivations = (
  seed: number,
): TransformerActivations => {
  return {
    transformerInput: [
      [seed, seed + 1],
      [seed + 2, seed + 3],
    ],
    attention: {
      normalizedInput: [
        [seed + 10, seed + 11],
        [seed + 12, seed + 13],
      ],
      heads: [],
      outMatrixInputActivations: [
        [seed + 20, seed + 21],
        [seed + 22, seed + 23],
      ],
      output: [
        [seed + 30, seed + 31],
        [seed + 32, seed + 33],
      ],
    },
    mlp: {
      normalizedInputToUpping: [
        [seed + 40, seed + 41],
        [seed + 42, seed + 43],
      ],
      uppingToNonLinear: [
        [seed + 50, seed + 51],
        [seed + 52, seed + 53],
      ],
      nonLinearToDowning: [
        [seed + 60, seed + 61],
        [seed + 62, seed + 63],
      ],
      downingOutput: [
        [seed + 70, seed + 71],
        [seed + 72, seed + 73],
      ],
    },
  };
};

describe("transformersBackprop", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("adds residual gradients around both transformer branches", () => {
    const outputGradients = [
      [10, 20],
      [30, 40],
    ];
    const weights = [makeTransformerWeights(1)];
    const activations = [makeTransformerActivations(100)];
    const mlpInputGradients = [
      [1, 2],
      [3, 4],
    ];
    const mlpNormInputGradients = [
      [5, 6],
      [7, 8],
    ];
    const attentionInputGradients = [
      [9, 10],
      [11, 12],
    ];
    const attentionNormInputGradients = [
      [13, 14],
      [15, 16],
    ];
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
    const outputGradients = [[1, 2]];
    const weights = [makeTransformerWeights(1), makeTransformerWeights(100)];
    const activations = [
      makeTransformerActivations(10),
      makeTransformerActivations(200),
    ];
    const layerOneMlpInputGradients = [[3, 4]];
    const layerOneMlpNormGradients = [[5, 6]];
    const layerOneAttentionInputGradients = [[7, 8]];
    const layerOneAttentionNormGradients = [[9, 10]];
    const layerZeroOutputGradients = addMatrices(
      addMatrices(outputGradients, layerOneMlpNormGradients),
      layerOneAttentionNormGradients,
    );
    const layerZeroMlpInputGradients = [[11, 12]];
    const layerZeroMlpNormGradients = [[13, 14]];
    const layerZeroAttentionInputGradients = [[15, 16]];
    const layerZeroAttentionNormGradients = [[17, 18]];
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
    const outputGradients = [[1, 2]];
    const weights = [makeTransformerWeights(1)];
    const activations = [makeTransformerActivations(100)];
    const expectedMlpNormalizationInput = addMatrices(
      activations[0]!.transformerInput,
      activations[0]!.attention.output,
    );

    mocks.backpropMlp.mockReturnValue({
      inputActivationGradients: [[3, 4]],
      weightGradients: makeMlpWeights(200),
    });
    mocks.backpropNormalize.mockReturnValueOnce([[5, 6]]).mockReturnValueOnce([
      [7, 8],
    ]);
    mocks.attentionBackprop.mockReturnValue({
      inputGradients: [[9, 10]],
      weightGradients: makeAttentionWeights(300),
    });

    transformersBackprop(outputGradients, weights, activations);

    expect(mocks.backpropNormalize.mock.calls[0]![1]).toEqual(
      expectedMlpNormalizationInput,
    );
  });
});
