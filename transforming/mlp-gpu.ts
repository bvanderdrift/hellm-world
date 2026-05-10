import { createMatrix } from "../shared/matrices.ts";
import {
  addVectorAcrossMatrixOnGPU,
  createMatrixBuffer,
  multiplyMatricesOnGPU,
  type MatrixBuffer,
} from "../shared/matrices-gpu.ts";
import type { MultilayerPerceptronGPUBuffers } from "../model/model-gpu-helpers.ts";
import { reluOnGpu } from "../shared/math-gpu.ts";

export const getMultilayerPerceptronActivationsOnGPU = (
  encoding: MatrixBuffer,
  perceptron: MultilayerPerceptronGPUBuffers,
  mlpScale: number,
) => {
  const upped = createMatrixBuffer(
    createMatrix(encoding.vectors, encoding.dimensions * mlpScale, () => 0),
  );

  multiplyMatricesOnGPU(encoding, perceptron.wUp.weightsMatrix, upped);

  addVectorAcrossMatrixOnGPU(upped, perceptron.wUp.biasVector);

  // We activate neurons
  reluOnGpu(upped);

  const out = createMatrixBuffer(
    createMatrix(encoding.vectors, encoding.dimensions, () => 0),
  );

  // We select new knowledge to enrich
  multiplyMatricesOnGPU(upped, perceptron.wDown.weightsMatrix, out);

  // Not sure what this bias does
  addVectorAcrossMatrixOnGPU(out, perceptron.wDown.biasVector);

  return out;
};
