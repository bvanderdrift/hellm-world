import { describe, it, expect } from "bun:test";
import { d } from "typegpu";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import { llmForwardPassByTokens } from "./llm.ts";
import { forwardPassOnGPU } from "./llm-gpu-forward-pass.ts";
import {
  loadWeightsIntoGpu,
  type WeightGPUBuffers,
} from "../model-gpu/model-weights-gpu.ts";
import { findTokenIndex } from "../model/model-helpers.ts";
import { getLatestCheckpointModel } from "../model/model-checkpoint-io.ts";
import { gpuContext } from "../shared/gpu-context.ts";
import { extractMatrixBuffer } from "../shared/matrices/matrices-gpu.ts";
import { tokenize } from "../shared/tokenizer.ts";
import {
  createMatrix,
  getRawVector,
  type Matrix,
} from "../shared/matrices/matrices.ts";
import { getHighestValueIndex, MAX_CONTEXT } from "./llm-shared.ts";
import { expectMatrixCloseTo } from "../testing/testing-utils.ts";
import type { Model } from "../model/model-types.ts";
import {
  allocateInferenceBuffers,
  type InferenceBuffers,
} from "../model-gpu/model-activations.gpu.ts";
import { emptyTrainingState as emptyHistory } from "../testing/model-fixtures.ts";

const destroyInferenceBuffers = (buffers: InferenceBuffers) => {
  for (const matrixBuffer of Object.values(buffers)) {
    matrixBuffer.buffer.destroy();
  }
};

const llmForwardPassByTokensOnGPU = async (
  input: string[],
  model: Model,
  weightBuffers: WeightGPUBuffers,
  withActivations: boolean,
): Promise<Matrix> => {
  const inferenceBuffers = allocateInferenceBuffers(input.length, 1, model);

  const inputPositionToVocabPosition = input.map((token) =>
    findTokenIndex(model.vocabulary, token),
  );

  const inputPositionToVocabPositionGPUBuffer = gpuContext
    .createBuffer(
      d.arrayOf(d.f32, inputPositionToVocabPosition.length),
      inputPositionToVocabPosition,
    )
    .$usage("storage");

  forwardPassOnGPU({
    weightBuffers,
    model,
    withActivations,
    inferenceBuffers,
    inputPositionToVocabPositionGPUBuffer,
  });

  const probabilities = await extractMatrixBuffer(
    inferenceBuffers.probabilitiesBuffer,
  );

  destroyInferenceBuffers(inferenceBuffers);
  inputPositionToVocabPositionGPUBuffer.buffer.destroy();

  return probabilities;
};

const seededRandom = (seed: number) => () => {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return (seed / 0x7fffffff) * 2 - 1;
};

type ModelConfig = {
  hidden: number;
  heads: number;
  layers: number;
  mlpMultiple: number;
  vocabSize: number;
};

const makeModel = (seed: number, config: ModelConfig): Model => {
  const { hidden, heads, layers, mlpMultiple, vocabSize } = config;
  const rand = seededRandom(seed);
  const randomMatrix = (vectors: number, dimensions: number): Matrix => ({
    vectors,
    dimensions,
    values: new Float32Array(
      Array.from({ length: vectors * dimensions }, rand),
    ),
  });

  return {
    name: "test-model",
    vocabulary: [
      ...Array.from({ length: vocabSize - 1 }, (_, i) => `t${i}`),
      END_OF_SEQUENCE_TOKEN,
    ],
    trainingState: emptyHistory,
    counts: {
      attentionHeads: heads,
      mlpMultiple,
      transformers: layers,
      hiddenDimensions: hidden,
    },
    embeddings: randomMatrix(vocabSize, hidden),
    unembeddings: randomMatrix(hidden, vocabSize),
    transformers: Array.from({ length: layers }, () => ({
      attention: {
        Q: randomMatrix(hidden, hidden),
        K: randomMatrix(hidden, hidden),
        V: randomMatrix(hidden, hidden),
        out: randomMatrix(hidden, hidden),
      },
      multilayerPerceptron: {
        wUp: {
          weightsMatrix: randomMatrix(hidden, hidden * mlpMultiple),
          biasVector: randomMatrix(1, hidden * mlpMultiple),
        },
        wDown: {
          weightsMatrix: randomMatrix(hidden * mlpMultiple, hidden),
          biasVector: randomMatrix(1, hidden),
        },
      },
    })),
  };
};

const softmaxRows = (input: Matrix): Matrix => {
  const output = createMatrix(input.vectors, input.dimensions);
  for (let row = 0; row < input.vectors; row++) {
    const start = row * input.dimensions;
    let biggest = -Infinity;
    for (let j = 0; j < input.dimensions; j++) {
      biggest = Math.max(biggest, input.values[start + j]!);
    }
    let summed = 0;
    for (let j = 0; j < input.dimensions; j++) {
      summed += Math.exp(input.values[start + j]! - biggest);
    }
    for (let j = 0; j < input.dimensions; j++) {
      output.values[start + j] =
        Math.exp(input.values[start + j]! - biggest) / summed;
    }
  }
  return output;
};

const cpuProbabilities = (input: string[], model: Model): Matrix =>
  softmaxRows(llmForwardPassByTokens(input, model, false).embeddings);

const argmaxLastRow = (probabilities: Matrix): number =>
  getHighestValueIndex(getRawVector(probabilities, probabilities.vectors - 1));

describe("forwardPassOnGPU matches the CPU forward pass", () => {
  const configs: { name: string; config: ModelConfig }[] = [
    {
      name: "single head, single layer",
      config: { hidden: 4, heads: 1, layers: 1, mlpMultiple: 1, vocabSize: 5 },
    },
    {
      name: "multi-head, single layer",
      config: { hidden: 8, heads: 2, layers: 1, mlpMultiple: 1, vocabSize: 5 },
    },
    {
      name: "single head, deep stack",
      config: { hidden: 8, heads: 1, layers: 4, mlpMultiple: 1, vocabSize: 5 },
    },
    {
      name: "multi-head, deep stack",
      config: { hidden: 8, heads: 2, layers: 4, mlpMultiple: 1, vocabSize: 5 },
    },
    {
      name: "wide MLP (mlpMultiple=2)",
      config: { hidden: 8, heads: 2, layers: 2, mlpMultiple: 2, vocabSize: 5 },
    },
    {
      name: "wide MLP (mlpMultiple=4)",
      config: { hidden: 16, heads: 4, layers: 2, mlpMultiple: 4, vocabSize: 6 },
    },
    {
      name: "addy-shaped (8 heads, 6 layers, mlp 4)",
      config: {
        hidden: 32,
        heads: 8,
        layers: 6,
        mlpMultiple: 4,
        vocabSize: 13,
      },
    },
  ];

  for (const { name, config } of configs) {
    it(name, async () => {
      const model = makeModel(101, config);
      const weightBuffers = loadWeightsIntoGpu(model);
      const input = ["t0", "t1", "t2", "t3"];

      const gpu = await llmForwardPassByTokensOnGPU(
        input,
        model,
        weightBuffers,
        false,
      );

      expectMatrixCloseTo(gpu, cpuProbabilities(input, model), 4);
    });
  }
});

describe("forwardPassOnGPU across context lengths", () => {
  const config: ModelConfig = {
    hidden: 16,
    heads: 4,
    layers: 3,
    mlpMultiple: 4,
    vocabSize: 6,
  };

  for (let length = 1; length <= 6; length++) {
    it(`context length ${length}`, async () => {
      const model = makeModel(202, config);
      const weightBuffers = loadWeightsIntoGpu(model);
      const input = Array.from(
        { length },
        (_, i) => `t${i % (config.vocabSize - 1)}`,
      );

      const gpu = await llmForwardPassByTokensOnGPU(
        input,
        model,
        weightBuffers,
        false,
      );

      expectMatrixCloseTo(gpu, cpuProbabilities(input, model), 4);
    });
  }
});

describe("forwardPassOnGPU keeps the residual un-normalized", () => {
  it("uses x + f(norm(x)) when x !== norm(x) (large embeddings)", async () => {
    const model = makeModel(303, {
      hidden: 8,
      heads: 2,
      layers: 2,
      mlpMultiple: 2,
      vocabSize: 5,
    });
    for (let i = 0; i < model.embeddings.values.length; i++) {
      model.embeddings.values[i] = model.embeddings.values[i]! * 12 + 8;
    }
    const weightBuffers = loadWeightsIntoGpu(model);
    const input = ["t0", "t1", "t2", "t3"];

    const gpu = await llmForwardPassByTokensOnGPU(
      input,
      model,
      weightBuffers,
      false,
    );

    expectMatrixCloseTo(gpu, cpuProbabilities(input, model), 4);
  });
});

describe("GPU and CPU greedy decoding agree token-for-token", () => {
  it("selects the same next token at every step of a multi-step rollout", async () => {
    const model = makeModel(404, {
      hidden: 16,
      heads: 4,
      layers: 3,
      mlpMultiple: 4,
      vocabSize: 6,
    });
    const weightBuffers = loadWeightsIntoGpu(model);

    let context = ["t0", "t1"];
    for (let step = 0; step < 6; step++) {
      const cpuIndex = argmaxLastRow(cpuProbabilities(context, model));
      const gpuIndex = argmaxLastRow(
        await llmForwardPassByTokensOnGPU(context, model, weightBuffers, false),
      );

      expect(gpuIndex).toBe(cpuIndex);

      context = [...context, model.vocabulary[cpuIndex]!];
    }
  });
});

const runBatchedForwardPassOnGPU = async (
  sequences: string[][],
  model: Model,
  weightBuffers: WeightGPUBuffers,
): Promise<Matrix> => {
  const inferenceBuffers = allocateInferenceBuffers(
    MAX_CONTEXT,
    sequences.length,
    model,
  );

  const inputPositionToVocabPosition = sequences.flatMap((tokens) =>
    new Array(MAX_CONTEXT).fill(0).map((_, tokenIndex) => {
      const token = tokens[tokenIndex];
      if (!token) return 0;
      return findTokenIndex(model.vocabulary, token);
    }),
  );

  const inputPositionToVocabPositionGPUBuffer = gpuContext
    .createBuffer(
      d.arrayOf(d.f32, sequences.length * MAX_CONTEXT),
      inputPositionToVocabPosition,
    )
    .$usage("storage");

  forwardPassOnGPU({
    weightBuffers,
    model,
    withActivations: false,
    inferenceBuffers,
    inputPositionToVocabPositionGPUBuffer,
  });

  const probabilities = await extractMatrixBuffer(
    inferenceBuffers.probabilitiesBuffer,
  );

  destroyInferenceBuffers(inferenceBuffers);
  inputPositionToVocabPositionGPUBuffer.buffer.destroy();

  return probabilities;
};

describe("forwardPassOnGPU — multi-batch", () => {
  it("matches the CPU probabilities for every sequence packed in one batch", async () => {
    const model = makeModel(505, {
      hidden: 16,
      heads: 4,
      layers: 3,
      mlpMultiple: 4,
      vocabSize: 6,
    });
    const weightBuffers = loadWeightsIntoGpu(model);

    const sequences = [["t0", "t1", "t2"], ["t1", "t2", "t3", "t4"], ["t0"]];

    const batched = await runBatchedForwardPassOnGPU(
      sequences,
      model,
      weightBuffers,
    );

    sequences.forEach((sequence, batchIndex) => {
      const expected = cpuProbabilities(sequence, model);
      for (let position = 0; position < sequence.length; position++) {
        const actualRow = getRawVector(
          batched,
          batchIndex * MAX_CONTEXT + position,
        );
        const expectedRow = getRawVector(expected, position);
        for (let token = 0; token < expectedRow.length; token++) {
          expect(actualRow[token]).toBeCloseTo(expectedRow[token]!, 4);
        }
      }
    });
  });
});

describe("real addy model regression", () => {
  it("reproduces the CPU answer for 500+510= on GPU", async () => {
    const model = getLatestCheckpointModel("addy");
    const weightBuffers = loadWeightsIntoGpu(model);
    const prompt = tokenize("500+510=", model.vocabulary);

    let cpuContext = [...prompt];
    let gpuContext = [...prompt];
    const cpuOut: string[] = [];
    const gpuOut: string[] = [];

    for (let step = 0; step < 6; step++) {
      const cpuToken =
        model.vocabulary[argmaxLastRow(cpuProbabilities(cpuContext, model))]!;
      const gpuToken =
        model.vocabulary[
          argmaxLastRow(
            await llmForwardPassByTokensOnGPU(
              gpuContext,
              model,
              weightBuffers,
              false,
            ),
          )
        ]!;

      if (
        cpuToken === END_OF_SEQUENCE_TOKEN &&
        gpuToken === END_OF_SEQUENCE_TOKEN
      ) {
        break;
      }

      cpuOut.push(cpuToken);
      gpuOut.push(gpuToken);
      cpuContext.push(cpuToken);
      gpuContext.push(gpuToken);
    }

    expect(gpuOut.join("")).toBe(cpuOut.join(""));
  });
});
