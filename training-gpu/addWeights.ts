import type { WeightGPUBuffers } from "../model-gpu/model-weights-gpu.ts";
import { addMatricesOnGPU } from "../shared/matrices/matrices-gpu.ts";

export const addWeights = (
  base: WeightGPUBuffers,
  adjustments: WeightGPUBuffers,
) => {
  addMatricesOnGPU(base.embeddings, adjustments.embeddings);
  for (let index = 0; index < base.transformers.length; index++) {
    const transformer = base.transformers[index];
    const transformerAdjustments = adjustments.transformers[index];

    if (!transformer) {
      throw new Error(`Failed to find transformer at index ${index}`);
    }

    if (!transformerAdjustments) {
      throw new Error(
        `Failed to find transformer adjustments at index ${index}`,
      );
    }

    addMatricesOnGPU(
      transformer.attention.Q,
      transformerAdjustments.attention.Q,
    );
    addMatricesOnGPU(
      transformer.attention.K,
      transformerAdjustments.attention.K,
    );
    addMatricesOnGPU(
      transformer.attention.V,
      transformerAdjustments.attention.V,
    );
    addMatricesOnGPU(
      transformer.attention.out,
      transformerAdjustments.attention.out,
    );

    addMatricesOnGPU(
      transformer.multilayerPerceptron.wUp.weightsMatrix,
      transformerAdjustments.multilayerPerceptron.wUp.weightsMatrix,
    );
    addMatricesOnGPU(
      transformer.multilayerPerceptron.wUp.biasVector,
      transformerAdjustments.multilayerPerceptron.wUp.biasVector,
    );

    addMatricesOnGPU(
      transformer.multilayerPerceptron.wDown.weightsMatrix,
      transformerAdjustments.multilayerPerceptron.wDown.weightsMatrix,
    );
    addMatricesOnGPU(
      transformer.multilayerPerceptron.wDown.biasVector,
      transformerAdjustments.multilayerPerceptron.wDown.biasVector,
    );
  }

  addMatricesOnGPU(base.unembeddings, adjustments.unembeddings);
};
