import { d, type TgpuBuffer, type UniformFlag } from "typegpu";
import { createMatrix, multiplyMatrices } from "../../shared/matrices.ts";
import { createMatrixBufferAndCopy } from "../../shared/matrices-gpu.ts";
import { gpuContext } from "../../shared/gpu-context.ts";
import { divideToWhole } from "../../shared/math.ts";
import { runSelfAttentionHead } from "./attention.ts";
import { runSelfAttentionMechanismOnGPU } from "./attention-gpu.ts";
import type { AttentionWeights } from "../../model/model-types.ts";
import type { AttentionGPUBuffers } from "../../model/model-gpu-helpers.ts";
import { compareAcrossSizes, rand } from "../../bench-harness.ts";

const HEAD_DIM = 32;

type AttentionCtx = {
  headsCount: number;
  headDimensionsCount: number;
  contextLength: TgpuBuffer<d.U32> & UniformFlag;
  headDimensions: TgpuBuffer<d.U32> & UniformFlag;
  weights: AttentionWeights;
  gpuWeights: AttentionGPUBuffers;
  scratch: {
    inputQ: ReturnType<typeof createMatrixBufferAndCopy>;
    inputK: ReturnType<typeof createMatrixBufferAndCopy>;
    inputV: ReturnType<typeof createMatrixBufferAndCopy>;
    attentionRelevancyOutput: ReturnType<typeof createMatrixBufferAndCopy>;
    matchingKeyProducts: ReturnType<typeof createMatrixBufferAndCopy>;
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
      const headsCount = divideToWhole(dimensions, HEAD_DIM);
      const scratchBuf = () =>
        createMatrixBufferAndCopy(createMatrix(vectors, dimensions));
      const relevancyBuf = () =>
        createMatrixBufferAndCopy(createMatrix(vectors, vectors * headsCount));
      return {
        headsCount,
        headDimensionsCount: HEAD_DIM,
        contextLength: gpuContext
          .createBuffer(d.u32, vectors)
          .$usage("uniform"),
        headDimensions: gpuContext
          .createBuffer(d.u32, HEAD_DIM)
          .$usage("uniform"),
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
          attentionRelevancyOutput: relevancyBuf(),
          matchingKeyProducts: relevancyBuf(),
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
    gpu: (
      { buffer },
      { headsCount, contextLength, headDimensions, gpuWeights, scratch },
    ) => {
      runSelfAttentionMechanismOnGPU(
        buffer,
        headsCount,
        contextLength,
        headDimensions,
        gpuWeights,
        scratch.inputQ,
        scratch.inputK,
        scratch.inputV,
        scratch.attentionRelevancyOutput,
        scratch.matchingKeyProducts,
        scratch.headsOut,
        scratch.attentionUpdate,
      );

      return scratch.attentionUpdate;
    },
  });
}
