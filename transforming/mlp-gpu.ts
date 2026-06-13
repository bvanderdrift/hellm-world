import {
  addVectorAcrossMatrixOnGPU,
  multiplyMatricesOnGPU,
  type MatrixBuffer,
} from "../shared/matrices-gpu.ts";
import type { MultilayerPerceptronGPUBuffers } from "../model/model-gpu-helpers.ts";
import { reluOnGpu } from "../shared/relu-gpu.ts";

export const getMultilayerPerceptronActivationsOnGPU = (
  encoding: MatrixBuffer,
  /** Pre-initialize buffer to work with, so that each MLP call doesn't have to allocate and remove GPU memory */
  upped: MatrixBuffer,
  /** Pre-initialize buffer to work with, so that each MLP call doesn't have to allocate and remove GPU memory */
  out: MatrixBuffer,
  perceptron: MultilayerPerceptronGPUBuffers,
  encoder: GPUCommandEncoder,
) => {
  multiplyMatricesOnGPU(encoding, perceptron.wUp.weightsMatrix, upped, encoder);

  addVectorAcrossMatrixOnGPU(upped, perceptron.wUp.biasVector, encoder);

  // We activate neurons
  reluOnGpu(upped, encoder);

  // We select new knowledge to enrich
  multiplyMatricesOnGPU(upped, perceptron.wDown.weightsMatrix, out, encoder);

  // Not sure what this bias does
  addVectorAcrossMatrixOnGPU(out, perceptron.wDown.biasVector, encoder);
};
