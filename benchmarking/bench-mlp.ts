import { createMatrix } from "../shared/matrices.ts";
import { createMatrixBufferAndCopy } from "../shared/matrices-gpu.ts";
import { getMultilayerPerceptronActivations } from "../transforming/mlp.ts";
import { getMultilayerPerceptronActivationsOnGPU } from "../transforming/mlp-gpu.ts";
import type { MultilayerPerceptronWeights } from "../model/model-types.ts";
import type { MultilayerPerceptronGPUBuffers } from "../model/model-gpu-helpers.ts";
import { compareAcrossSizes, rand } from "./bench-harness.ts";

const MLP_MULTIPLE = 4;

type MlpCtx = {
  weights: MultilayerPerceptronWeights;
  gpu: MultilayerPerceptronGPUBuffers;
  upped: ReturnType<typeof createMatrixBufferAndCopy>;
  out: ReturnType<typeof createMatrixBufferAndCopy>;
};

if (import.meta.main) {
  console.log(`  mlpMultiple=${MLP_MULTIPLE}`);
  await compareAcrossSizes<MlpCtx>({
    name: "MLP CPU vs GPU benchmark",
    setup: ({ vectors, dimensions }) => {
      const weights: MultilayerPerceptronWeights = {
        wUp: {
          weightsMatrix: createMatrix(dimensions, dimensions * MLP_MULTIPLE, rand),
          biasVector: createMatrix(1, dimensions * MLP_MULTIPLE, rand),
        },
        wDown: {
          weightsMatrix: createMatrix(dimensions * MLP_MULTIPLE, dimensions, rand),
          biasVector: createMatrix(1, dimensions, rand),
        },
      };
      return {
        weights,
        gpu: {
          wUp: {
            weightsMatrix: createMatrixBufferAndCopy(weights.wUp.weightsMatrix),
            biasVector: createMatrixBufferAndCopy(weights.wUp.biasVector),
          },
          wDown: {
            weightsMatrix: createMatrixBufferAndCopy(weights.wDown.weightsMatrix),
            biasVector: createMatrixBufferAndCopy(weights.wDown.biasVector),
          },
        },
        upped: createMatrixBufferAndCopy(
          createMatrix(vectors, dimensions * MLP_MULTIPLE),
        ),
        out: createMatrixBufferAndCopy(createMatrix(vectors, dimensions)),
      };
    },
    cpu: ({ matrix }, { weights }) =>
      getMultilayerPerceptronActivations(matrix, weights).downingOutput,
    gpu: ({ buffer }, { gpu, upped, out }) => {
      getMultilayerPerceptronActivationsOnGPU(buffer, upped, out, gpu);
      return out;
    },
  });
}
