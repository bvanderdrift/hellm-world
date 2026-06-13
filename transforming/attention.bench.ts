import { createMatrix, multiplyMatrices } from "../shared/matrices.ts";
import { createMatrixBufferAndCopy } from "../shared/matrices-gpu.ts";
import { divideToWhole } from "../shared/math.ts";
import { runSelfAttentionHead } from "./attention.ts";
import { runSelfAttentionMechanismOnGPU } from "./attention-gpu.ts";
import type { AttentionWeights } from "../model/model-types.ts";
import type { AttentionGPUBuffers } from "../model/model-gpu-helpers.ts";
import { compareAcrossSizes, rand } from "../bench-harness.ts";

const HEAD_DIM = 32;

type AttentionCtx = {
  headsCount: number;
  headDimensionsCount: number;
  weights: AttentionWeights;
  gpuWeights: AttentionGPUBuffers;
  scratch: {
    inputQ: ReturnType<typeof createMatrixBufferAndCopy>;
    inputK: ReturnType<typeof createMatrixBufferAndCopy>;
    inputV: ReturnType<typeof createMatrixBufferAndCopy>;
    attentionUpdate: ReturnType<typeof createMatrixBufferAndCopy>;
    headsOut: ReturnType<typeof createMatrixBufferAndCopy>;
  };
};

if (import.meta.main) {
  await compareAcrossSizes<AttentionCtx>({
    name: "Attention mechanism CPU vs GPU benchmark",
    setup: ({ vectors, dimensions }) => {
      const square = () => createMatrix(dimensions, dimensions, rand);
      const weights: AttentionWeights = {
        Q: square(),
        K: square(),
        V: square(),
        out: square(),
      };
      const scratchBuf = () =>
        createMatrixBufferAndCopy(createMatrix(vectors, dimensions));
      return {
        headsCount: divideToWhole(dimensions, HEAD_DIM),
        headDimensionsCount: HEAD_DIM,
        weights,
        gpuWeights: {
          Q: createMatrixBufferAndCopy(weights.Q),
          K: createMatrixBufferAndCopy(weights.K),
          V: createMatrixBufferAndCopy(weights.V),
          out: createMatrixBufferAndCopy(weights.out),
        },
        scratch: {
          inputQ: scratchBuf(),
          inputK: scratchBuf(),
          inputV: scratchBuf(),
          attentionUpdate: scratchBuf(),
          headsOut: scratchBuf(),
        },
      };
    },
    cpu: ({ matrix }, { weights, headsCount, headDimensionsCount }) => {
      const head = runSelfAttentionHead(
        multiplyMatrices(matrix, weights.Q),
        multiplyMatrices(matrix, weights.K),
        multiplyMatrices(matrix, weights.V),
        headsCount,
        headDimensionsCount,
      );
      return multiplyMatrices(head.output, weights.out);
    },
    gpu: ({ buffer }, { headsCount, gpuWeights, scratch }) =>
      runSelfAttentionMechanismOnGPU(
        buffer,
        headsCount,
        gpuWeights,
        scratch.inputQ,
        scratch.inputK,
        scratch.inputV,
        scratch.attentionUpdate,
        scratch.headsOut,
      ).output,
  });
}
