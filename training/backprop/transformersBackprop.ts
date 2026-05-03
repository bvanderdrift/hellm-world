import type { TransformerActivations } from "../../model/activations-types.ts";
import type { TransformerWeights } from "../../model/model-types.ts";
import { addMatrices } from "../../shared/matrices.ts";
import { attentionBackprop } from "./attentionBackprop.ts";
import { backpropMlp } from "./mlpBackprop.ts";
import { backpropNormalize } from "./normalizeBackprop.ts";

export const transformersBackprop = (
  outputGradients: number[][],
  weights: TransformerWeights[],
  activations: TransformerActivations[],
): {
  transformerGradients: TransformerWeights[];
  inputActivationGradients: number[][];
} => {
  if (weights.length !== activations.length) {
    throw new Error(
      `Transformer weights count ${weights.length} does not equal transformers activations count ${activations.length}`,
    );
  }

  // Spread b/c reverse mutates in-place
  const reversedActivations = [...activations].reverse();

  let lastOutputGradients = outputGradients;
  let transformerGradients: TransformerWeights[] = [];

  for (let index = 0; index < reversedActivations.length; index++) {
    const transformerActivations = reversedActivations[index]!;
    const transformerWeights = weights[weights.length - index - 1]!;

    const {
      inputActivationGradients: mlpInputGradients,
      weightGradients: mlpWeightGradients,
    } = backpropMlp(
      transformerWeights.multilayerPerceptron,
      transformerActivations.mlp,
      /**
       * z_i = h_i + m_i
       * So to determine dL/dm_i we just need dL/dz_i which is lastOutputGradients
       */
      lastOutputGradients,
    );

    const preNormalizationInputGradients = backpropNormalize(
      mlpInputGradients,
      addMatrices(
        transformerActivations.attention.output,
        transformerActivations.transformerInput,
      ),
    );

    const attentionOutputGradients = combineGradients(
      lastOutputGradients,
      preNormalizationInputGradients,
    );

    const {
      inputGradients: attentionInputGradients,
      weightGradients: attentionWeightGradients,
    } = attentionBackprop(
      transformerWeights.attention,
      attentionOutputGradients,
      transformerActivations.attention,
    );

    const preAttentionNormInputGradients = backpropNormalize(
      attentionInputGradients,
      transformerActivations.transformerInput,
    );

    const transformerInputGradients = combineGradients(
      attentionOutputGradients,
      preAttentionNormInputGradients,
    );

    // We traverse in reverse and we want to add in correct order, so we unshift instead of push
    transformerGradients.unshift({
      attention: attentionWeightGradients,
      multilayerPerceptron: mlpWeightGradients,
    });

    lastOutputGradients = transformerInputGradients;
  }

  return {
    inputActivationGradients: lastOutputGradients,
    transformerGradients,
  };
};

/**
 * z_i = h_i + b_i where h_i is residual embedding and b_i is update values from branch.
 *
 * dL/dh_i = dL/dz_i * dz_i/dh_i + dL/db_i becaude b_i is a product of h_i
 *  = dL/dz_i * 1 + combinedOutputGradients_i
 */
const combineGradients = (
  combinedOutputGradients: number[][],
  branchInputGradients: number[][],
) =>
  combinedOutputGradients.map((outputGradientVector, vectorIndex) => {
    return outputGradientVector.map((dLdz_i, dimensionIndex) => {
      return dLdz_i + branchInputGradients[vectorIndex]![dimensionIndex]!;
    });
  });
