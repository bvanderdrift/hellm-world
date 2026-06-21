/**
 * Correctness testing for the `addy-new` addition model.
 *
 * Given a checkpoint version number it loads that exact checkpoint onto the GPU
 * and forever samples random batches of pairs (a, b) with a, b < 10_000. Each
 * batch is tested in a single batched GPU forward pass: rather than
 * autoregressively generating, it teacher-forces the full sequence
 * "a+b=<answer>" and checks, at every answer position, whether the model's
 * argmax prediction equals the expected next token. That is equivalent to a
 * fully-correct greedy decode but needs only one forward pass per batch.
 *
 * This root process owns the single grid file (`correctness_grid.bin`): it folds
 * every result in and, every 1000 tests, saves the grid and dumps a heatmap PNG
 * via the charting module so accuracy can be inspected visually.
 *
 * The charting half lives in `correctness-chart.ts` and can be run on its own
 * to (re)render an already-saved grid without retesting.
 *
 * This is an independent script: it only imports already-exported inference
 * helpers, it does not modify any inference/training code.
 *
 * Usage: bun run models/addy-new/scripts/correctness-map.ts <checkpointId> [-b <batchSize>]
 */

import { d } from "typegpu";
import {
  GRID_SIZE,
  MODEL_NAME,
  TRUE,
  FALSE,
  gridPath,
  loadGrid,
  saveGrid,
} from "./correctness-grid.ts";
import { dumpImage } from "./correctness-chart.ts";
import { getCheckpointModel } from "../../../model/model-checkpoint-io.ts";
import {
  loadWeightsIntoGpu,
  type WeightGPUBuffers,
} from "../../../model-gpu/model-weights-gpu.ts";
import { findTokenIndex } from "../../../model/model-helpers.ts";
import type { Model } from "../../../model/model-types.ts";
import { forwardPassOnGPU } from "../../../running/llm-gpu-forward-pass.ts";
import {
  MAX_CONTEXT,
  getHighestValueIndex,
} from "../../../running/llm-shared.ts";
import { END_OF_SEQUENCE_TOKEN } from "../../../shared/const.ts";
import { gpuContext } from "../../../shared/gpu-context.ts";
import { extractMatrixBuffer } from "../../../shared/matrices/matrices-gpu.ts";
import { getRawVector } from "../../../shared/matrices/matrices.ts";
import { tokenize } from "../../../shared/tokenizer.ts";
import {
  allocateInferenceBuffers,
  type InferenceBuffers,
} from "../../../model-gpu/model-activations.gpu.ts";

type Result = { a: number; b: number; correct: boolean };

const DUMP_EVERY = 1000;

const PROGRESS_WIDTH = 30;

/** In-place progress bar towards the next render, e.g. "[------        ]  37%". */
const writeProgress = (fraction: number) => {
  const filled = Math.round(fraction * PROGRESS_WIDTH);
  const bar = "-".repeat(filled) + " ".repeat(PROGRESS_WIDTH - filled);
  const pct = Math.round(fraction * 100)
    .toString()
    .padStart(3);
  process.stdout.write(`\r[${bar}] ${pct}%`);
};

/** Parse `<checkpointId> [-b <batchSize>]` from argv; `-b` defaults to 256. */
const parseArgs = (): { checkpointId: number; batchSize: number } => {
  const args = process.argv.slice(2);
  let checkpointArg: string | undefined;
  let batchSize = 256;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "-b") {
      batchSize = Number(args[++i]);
    } else {
      checkpointArg = args[i];
    }
  }

  if (
    checkpointArg === undefined ||
    Number.isNaN(Number(checkpointArg)) ||
    !Number.isInteger(batchSize) ||
    batchSize < 1
  ) {
    console.error(
      "Usage: bun run models/addy-new/scripts/correctness-map.ts <checkpointId> [-b <batchSize>]",
    );
    process.exit(1);
  }

  return { checkpointId: Number(checkpointArg), batchSize };
};

/**
 * Teacher-forces a whole batch of pairs in one GPU forward pass and returns,
 * per pair, whether every answer-position prediction matched the expected token.
 */
const checkBatch = async (
  pairs: { a: number; b: number }[],
  model: Model,
  weightBuffers: WeightGPUBuffers,
  inferenceBuffers: InferenceBuffers,
): Promise<Result[]> => {
  const checks = pairs.map(({ a, b }) => {
    const promptTokens = tokenize(`${a}+${b}=`, model.vocabulary);
    const answerTokens = tokenize(String(a + b), model.vocabulary);
    const inputTokens = [...promptTokens, ...answerTokens];

    const expected = [...answerTokens, END_OF_SEQUENCE_TOKEN].map((token) =>
      findTokenIndex(model.vocabulary, token),
    );

    return {
      firstCheckPosition: promptTokens.length - 1,
      inputTokens,
      expected,
    };
  });

  const inputPositionToVocabPosition = checks.flatMap(({ inputTokens }) =>
    new Array(MAX_CONTEXT).fill(0).map((_, tokenIndex) => {
      const token = inputTokens[tokenIndex];
      if (!token) return 0;
      return findTokenIndex(model.vocabulary, token);
    }),
  );

  const inputPositionToVocabPositionGPUBuffer = gpuContext
    .createBuffer(
      d.arrayOf(d.f32, pairs.length * MAX_CONTEXT),
      inputPositionToVocabPosition,
    )
    .$usage("storage");

  const tDispatchStart = performance.now();

  forwardPassOnGPU({
    weightBuffers,
    model,
    withActivations: false,
    inferenceBuffers,
    inputPositionToVocabPositionGPUBuffer,
  });

  await gpuContext.device.queue.onSubmittedWorkDone();
  const tGpuDone = performance.now();

  const probabilities = await extractMatrixBuffer(
    inferenceBuffers.probabilitiesBuffer,
  );
  const tReadbackDone = performance.now();

  inputPositionToVocabPositionGPUBuffer.buffer.destroy();

  const results = pairs.map(({ a, b }, batchIndex) => {
    const { firstCheckPosition, expected } = checks[batchIndex]!;

    const correct = expected.every((expectedIndex, step) => {
      const row = batchIndex * MAX_CONTEXT + firstCheckPosition + step;
      const predicted = getHighestValueIndex(getRawVector(probabilities, row));
      return predicted === expectedIndex;
    });

    return { a, b, correct };
  });
  const tArgmaxDone = performance.now();

  const gpuMs = tGpuDone - tDispatchStart;
  const readbackMs = tReadbackDone - tGpuDone;
  const argmaxMs = tArgmaxDone - tReadbackDone;
  const totalMs = tArgmaxDone - tDispatchStart;
  const tokensPerSec = (pairs.length * MAX_CONTEXT) / (totalMs / 1000);
  console.log(
    `\n[timing] batch=${pairs.length} · gpu=${gpuMs.toFixed(1)}ms (${((gpuMs / totalMs) * 100).toFixed(0)}%) · readback=${readbackMs.toFixed(1)}ms (${((readbackMs / totalMs) * 100).toFixed(0)}%) · argmax=${argmaxMs.toFixed(1)}ms (${((argmaxMs / totalMs) * 100).toFixed(0)}%) · total=${totalMs.toFixed(1)}ms · ${Math.round(tokensPerSec).toLocaleString()} tok/s\n`,
  );

  return results;
};

const main = async () => {
  const { checkpointId, batchSize } = parseArgs();

  const {
    grid,
    testCount: resumedTestCount,
    correctCount: resumedCorrectCount,
  } = loadGrid(checkpointId);
  let testCount = resumedTestCount;
  let correctCount = resumedCorrectCount;

  if (testCount > 0) {
    console.log(
      `Resuming from ${gridPath(checkpointId)} — ${testCount.toLocaleString()} cells already tested.`,
    );
  }

  console.log(
    `Loading ${MODEL_NAME} checkpoint ${checkpointId} onto the GPU (batch size ${batchSize})...`,
  );
  console.log("Sampling forever — Ctrl-C to stop.\n");

  const model = getCheckpointModel(MODEL_NAME, checkpointId);
  const weightBuffers = loadWeightsIntoGpu(model.counts, model);
  const inferenceBuffers = allocateInferenceBuffers(
    MAX_CONTEXT,
    batchSize,
    model,
  );

  let lastDumpTime = performance.now();

  const onResults = async (results: Result[]) => {
    for (const { a, b, correct } of results) {
      // A cell may be re-sampled; only count genuinely new tests so
      // testCount/correctCount stay in sync with the grid's non-null cells
      // (matching how loadGrid derives them).
      const previous = grid[a * GRID_SIZE + b]!;
      grid[a * GRID_SIZE + b] = correct ? TRUE : FALSE;

      if (previous === TRUE) correctCount--;
      else if (previous === FALSE) {
        // already counted as a test, nothing to add
      } else {
        testCount++;
      }
      if (correct) correctCount++;

      const intoBatch = testCount % DUMP_EVERY;
      if (intoBatch === 0) {
        const now = performance.now();
        const dumpDurationMs = now - lastDumpTime;
        lastDumpTime = now;

        writeProgress(1);
        process.stdout.write("\n");

        saveGrid(grid, checkpointId);

        const outputPath = await dumpImage(
          grid,
          checkpointId,
          testCount,
          correctCount,
        );
        const accuracy = (correctCount / testCount) * 100;
        console.log(
          `${testCount.toLocaleString()} tests · ${accuracy.toFixed(2)}% correct · last ${DUMP_EVERY} in ${(dumpDurationMs / 1000).toFixed(2)}s · wrote ${outputPath}`,
        );
        process.stdout.write("\n");
      } else {
        writeProgress(intoBatch / DUMP_EVERY);
      }
    }
  };

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const pairs = new Array(batchSize).fill(0).map(() => ({
      a: Math.floor(Math.random() * GRID_SIZE),
      b: Math.floor(Math.random() * GRID_SIZE),
    }));

    const results = await checkBatch(
      pairs,
      model,
      weightBuffers,
      inferenceBuffers,
    );
    await onResults(results);
  }
};

main();
