import { d, type TgpuBuffer, type UniformFlag } from "typegpu";
import {
  createMatrix,
  multiplyMatrices,
  type Matrix,
} from "../../shared/matrices/matrices.ts";
import { createMatrixBuffer } from "../../shared/matrices/matrices-gpu.ts";
import { gpuContext } from "../../shared/gpu-context.ts";
import { divideToWhole } from "../../shared/math.ts";
import { runSelfAttentionHead } from "./attention.ts";
import { runSelfAttentionMechanismOnGPU } from "./attention-gpu.ts";
import type { AttentionWeights } from "../../model/model-types.ts";
import type { AttentionGPUBuffers } from "../../model-gpu/model-weights-gpu.ts";
import { compareAcrossSizes, rand } from "../../bench-harness.ts";
import { MAX_CONTEXT } from "../../running/llm-shared.ts";

const HEAD_DIM = 32;

const sliceRows = (matrix: Matrix, start: number, count: number): Matrix => {
  const slice = createMatrix(count, matrix.dimensions);
  slice.values.set(
    matrix.values.subarray(
      start * matrix.dimensions,
      (start + count) * matrix.dimensions,
    ),
  );
  return slice;
};

type AttentionCtx = {
  headsCount: number;
  headDimensionsCount: number;
  headDimensions: TgpuBuffer<d.U32> & UniformFlag;
  weights: AttentionWeights;
  gpuWeights: AttentionGPUBuffers;
  scratch: {
    inputQ: ReturnType<typeof createMatrixBuffer>;
    inputK: ReturnType<typeof createMatrixBuffer>;
    inputV: ReturnType<typeof createMatrixBuffer>;
    attentionRelevancyOutput: ReturnType<typeof createMatrixBuffer>;
    matchingKeyProducts: ReturnType<typeof createMatrixBuffer>;
    attentionUpdate: ReturnType<typeof createMatrixBuffer>;
    headsOut: ReturnType<typeof createMatrixBuffer>;
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
        createMatrixBuffer(createMatrix(vectors, dimensions));
      const relevancyBuf = () =>
        createMatrixBuffer(createMatrix(vectors, vectors * headsCount));
      return {
        headsCount,
        headDimensionsCount: HEAD_DIM,
        headDimensions: gpuContext
          .createBuffer(d.u32, HEAD_DIM)
          .$usage("uniform"),
        weights,
        gpuWeights: {
          Q: createMatrixBuffer(weights.Q),
          K: createMatrixBuffer(weights.K),
          V: createMatrixBuffer(weights.V),
          out: createMatrixBuffer(weights.out),
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
      const inputQ = multiplyMatrices(matrix, weights.Q);
      const inputK = multiplyMatrices(matrix, weights.K);
      const inputV = multiplyMatrices(matrix, weights.V);

      const headsOutput = createMatrix(matrix.vectors, matrix.dimensions);
      for (let start = 0; start < matrix.vectors; start += MAX_CONTEXT) {
        const count = Math.min(MAX_CONTEXT, matrix.vectors - start);
        const head = runSelfAttentionHead(
          sliceRows(inputQ, start, count),
          sliceRows(inputK, start, count),
          sliceRows(inputV, start, count),
          headsCount,
          headDimensionsCount,
        );
        headsOutput.values.set(head.output.values, start * matrix.dimensions);
      }

      return multiplyMatrices(headsOutput, weights.out);
    },
    gpu: ({ buffer }, { headsCount, headDimensions, gpuWeights, scratch }) => {
      runSelfAttentionMechanismOnGPU(
        buffer,
        headsCount,
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
