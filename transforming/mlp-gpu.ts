import {
  addVectorAcrossMatrixOnGPU,
  multiplyMatricesOnGPU,
  type MatrixBuffer,
} from "../shared/matrices-gpu.ts";
import type { MultilayerPerceptronGPUBuffers } from "../model/model-gpu-helpers.ts";
import { reluOnGpu } from "../shared/math-gpu.ts";

export const getMultilayerPerceptronActivationsOnGPU = (
  encoding: MatrixBuffer,
  /** Pre-initialize buffer to work with, so that each MLP call doesn't have to allocate and remove GPU memory */
  upped: MatrixBuffer,
  /** Pre-initialize buffer to work with, so that each MLP call doesn't have to allocate and remove GPU memory */
  out: MatrixBuffer,
  perceptron: MultilayerPerceptronGPUBuffers,
) => {
  multiplyMatricesOnGPU(encoding, perceptron.wUp.weightsMatrix, upped);

  addVectorAcrossMatrixOnGPU(upped, perceptron.wUp.biasVector);

  // We activate neurons
  reluOnGpu(upped);

  // We select new knowledge to enrich
  multiplyMatricesOnGPU(upped, perceptron.wDown.weightsMatrix, out);

  // Not sure what this bias does
  addVectorAcrossMatrixOnGPU(out, perceptron.wDown.biasVector);
};
