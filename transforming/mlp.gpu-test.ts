import { describe, it } from "bun:test";
import { gpuContext } from "../shared/gpu-context.ts";
import {
  createMatrixBuffer,
  extractMatrixBuffer,
} from "../shared/matrices-gpu.ts";
import type { Matrix } from "../shared/matrices.ts";
import type { MultilayerPerceptronWeights } from "../model/model-types.ts";
import type { MultilayerPerceptronGPUBuffers } from "../model/model-gpu-helpers.ts";
import {
  matrixFrom,
  vectorFrom,
  expectMatrixCloseTo,
} from "../testing/testing-utils.ts";
import { getMultilayerPerceptronActivations } from "./mlp.ts";
import { getMultilayerPerceptronActivationsOnGPU } from "./mlp-gpu.ts";

const loadPerceptron = (
  perceptron: MultilayerPerceptronWeights,
): MultilayerPerceptronGPUBuffers => ({
  wUp: {
    weightsMatrix: createMatrixBuffer(perceptron.wUp.weightsMatrix),
    biasVector: createMatrixBuffer(perceptron.wUp.biasVector),
  },
  wDown: {
    weightsMatrix: createMatrixBuffer(perceptron.wDown.weightsMatrix),
    biasVector: createMatrixBuffer(perceptron.wDown.biasVector),
  },
});

const runGpu = async (
  input: Matrix,
  perceptron: MultilayerPerceptronWeights,
): Promise<Matrix> => {
  const intermediateDim = perceptron.wUp.weightsMatrix.dimensions;
  const outDim = perceptron.wDown.weightsMatrix.dimensions;

  const encoding = createMatrixBuffer(input);
  const upped = createMatrixBuffer({
    vectors: input.vectors,
    dimensions: intermediateDim,
  });
  const out = createMatrixBuffer({
    vectors: input.vectors,
    dimensions: outDim,
  });

  const encoder = gpuContext.device.createCommandEncoder();
  getMultilayerPerceptronActivationsOnGPU(
    encoding,
    upped,
    out,
    loadPerceptron(perceptron),
    encoder,
  );
  gpuContext.device.queue.submit([encoder.finish()]);
  await gpuContext.device.queue.onSubmittedWorkDone();

  return extractMatrixBuffer(out);
};

const twoDimPerceptron: MultilayerPerceptronWeights = {
  wUp: {
    weightsMatrix: matrixFrom([
      [1, 0, -1, 2, 0, 1, 0, -2],
      [0, 1, 2, -1, 3, 0, -2, 1],
    ]),
    biasVector: vectorFrom([0, -1, 1, 0, -2, 2, 1, -3]),
  },
  wDown: {
    weightsMatrix: matrixFrom([
      [1, 0],
      [2, 1],
      [0, 3],
      [-1, 2],
      [4, 0],
      [0, -2],
      [1, 1],
      [3, 0],
    ]),
    biasVector: vectorFrom([0.5, -1.5]),
  },
};

describe("getMultilayerPerceptronActivationsOnGPU", () => {
  it("matches the CPU reference for a single row vector", async () => {
    const input = matrixFrom([[2, -1]]);

    const actual = await runGpu(input, twoDimPerceptron);
    const expected = getMultilayerPerceptronActivations(
      input,
      twoDimPerceptron,
    ).downingOutput;

    expectMatrixCloseTo(actual, expected, 4);
  });

  it("runs the same MLP independently on each row vector", async () => {
    const input = matrixFrom([
      [2, -1],
      [-1, 3],
      [0, 0],
    ]);

    const actual = await runGpu(input, twoDimPerceptron);
    const expected = getMultilayerPerceptronActivations(
      input,
      twoDimPerceptron,
    ).downingOutput;

    expectMatrixCloseTo(actual, expected, 4);
  });

  it("matches the CPU reference for a larger random case", async () => {
    const inputDim = 6;
    const intermediateDim = 32;
    const outDim = 6;
    const rand = () => Math.random() * 2 - 1;
    const randomMatrix = (vectors: number, dimensions: number): Matrix => ({
      vectors,
      dimensions,
      values: new Float32Array(
        Array.from({ length: vectors * dimensions }, rand),
      ),
    });

    const perceptron: MultilayerPerceptronWeights = {
      wUp: {
        weightsMatrix: randomMatrix(inputDim, intermediateDim),
        biasVector: randomMatrix(1, intermediateDim),
      },
      wDown: {
        weightsMatrix: randomMatrix(intermediateDim, outDim),
        biasVector: randomMatrix(1, outDim),
      },
    };
    const input = randomMatrix(5, inputDim);

    const actual = await runGpu(input, perceptron);
    const expected = getMultilayerPerceptronActivations(
      input,
      perceptron,
    ).downingOutput;

    expectMatrixCloseTo(actual, expected, 4);
  });
});
