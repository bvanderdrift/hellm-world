import { createMatrix } from "../shared/matrices/matrices.ts";
import { createMatrixBuffer } from "../shared/matrices/matrices-gpu.ts";
import { getMultilayerPerceptronActivations } from "./mlp.ts";
import { getMultilayerPerceptronActivationsOnGPU } from "./mlp-gpu.ts";
import type { MultilayerPerceptronWeights } from "../model/model-types.ts";
import type { MultilayerPerceptronGPUBuffers } from "../model-gpu/model-weights-gpu.ts";
import { compareAcrossSizes, rand } from "../bench-harness.ts";

const MLP_MULTIPLE = 4;

type MlpCtx = {
  weights: MultilayerPerceptronWeights;
  gpu: MultilayerPerceptronGPUBuffers;
  upped: ReturnType<typeof createMatrixBuffer>;
  out: ReturnType<typeof createMatrixBuffer>;
};

if (import.meta.main) {
  console.log(`  mlpMultiple=${MLP_MULTIPLE}`);
  await compareAcrossSizes<MlpCtx>({
    name: "MLP CPU vs GPU benchmark",
    setup: ({ vectors, dimensions }) => {
      const weights: MultilayerPerceptronWeights = {
        wUp: {
          weightsMatrix: createMatrix(
            dimensions,
            dimensions * MLP_MULTIPLE,
            rand,
          ),
          biasVector: createMatrix(1, dimensions * MLP_MULTIPLE, rand),
        },
        wDown: {
          weightsMatrix: createMatrix(
            dimensions * MLP_MULTIPLE,
            dimensions,
            rand,
          ),
          biasVector: createMatrix(1, dimensions, rand),
        },
      };
      return {
        weights,
        gpu: {
          wUp: {
            weightsMatrix: createMatrixBuffer(weights.wUp.weightsMatrix),
            biasVector: createMatrixBuffer(weights.wUp.biasVector),
          },
          wDown: {
            weightsMatrix: createMatrixBuffer(weights.wDown.weightsMatrix),
            biasVector: createMatrixBuffer(weights.wDown.biasVector),
          },
        },
        upped: createMatrixBuffer(
          createMatrix(vectors, dimensions * MLP_MULTIPLE),
        ),
        out: createMatrixBuffer(createMatrix(vectors, dimensions)),
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
