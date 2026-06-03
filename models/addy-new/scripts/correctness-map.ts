/**
 * Correctness testing for the `addy-new` addition model.
 *
 * Given a checkpoint version number it loads that exact checkpoint, then forever
 * samples random pairs (a, b) with a, b < 10_000, runs the model on "a+b=" and
 * records whether the fully-correct answer is produced into a single grid file
 * (`correctness_grid.bin`). Every 1000 tests it saves the grid and dumps a
 * heatmap PNG via the charting module so accuracy can be inspected visually.
 *
 * The charting half lives in `correctness-chart.ts` and can be run on its own
 * to (re)render an already-saved grid without retesting.
 *
 * This is an independent script: it only imports already-exported inference
 * helpers, it does not modify any inference/training code.
 *
 * Usage: bun run models/addy-new/scripts/correctness-map.ts <checkpointId>
 */

import { type Model } from "../../../model/model-types.ts";
import { tokenize } from "../../../shared/tokenizer.ts";
import { getRawVector } from "../../../shared/matrices.ts";
import { END_OF_SEQUENCE_TOKEN } from "../../../shared/const.ts";
import {
  getHighestValueIndex,
  llmForwardPassByTokens,
} from "../../../running/llm.ts";
import { getCheckpointModel } from "../../../model/model-checkpoint-io.ts";

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

const DUMP_EVERY = 1000;

/** Greedy argmax of the next-token logits for a given token sequence. */
const nextToken = (tokens: string[], model: Model): string => {
  const { embeddings: unembeddedState } = llmForwardPassByTokens(
    tokens,
    model,
    false,
  );
  const logits = getRawVector(unembeddedState, unembeddedState.vectors - 1);
  const index = getHighestValueIndex(logits);
  const token = model.vocabulary[index];
  if (!token) {
    throw new Error(`Failed to find token at index ${index}`);
  }
  return token;
};

/**
 * Returns true iff the model produces exactly `expected` digits followed by
 * <EOS>. Stops generating as soon as a token diverges from the expected answer.
 */
const isCorrect = (a: number, b: number, model: Model): boolean => {
  const input = `${a}+${b}=`;
  const inputTokens = tokenize(input, model.vocabulary);
  const expected = String(a + b);

  const generated: string[] = [];

  // expected.length digit tokens, then one <EOS>. Small cap guards runaways.
  const maxTokens = expected.length + 1;

  for (let step = 0; step < maxTokens; step++) {
    const token = nextToken([...inputTokens, ...generated], model);

    if (step < expected.length) {
      // Still expecting a digit; it must match exactly.
      if (token !== expected[step]) {
        return false;
      }
    } else {
      // All digits matched; the sequence must now terminate.
      return token === END_OF_SEQUENCE_TOKEN;
    }

    generated.push(token);
  }

  return false;
};

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

const main = async () => {
  const checkpointArg = process.argv[2];
  if (checkpointArg === undefined || Number.isNaN(Number(checkpointArg))) {
    console.error(
      "Usage: bun run models/addy-new/scripts/correctness-map.ts <checkpointId>",
    );
    process.exit(1);
  }
  const checkpointId = Number(checkpointArg);

  console.log(`Loading ${MODEL_NAME} checkpoint ${checkpointId}...`);
  const model = getCheckpointModel(MODEL_NAME, checkpointId);

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

  console.log("Sampling forever — Ctrl-C to stop.\n");

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const a = Math.floor(Math.random() * GRID_SIZE);
    const b = Math.floor(Math.random() * GRID_SIZE);

    const correct = isCorrect(a, b, model);
    grid[a * GRID_SIZE + b] = correct ? TRUE : FALSE;

    testCount++;
    if (correct) correctCount++;

    const intoBatch = testCount % DUMP_EVERY;
    if (intoBatch === 0) {
      // Batch complete: finish the bar, render, then start a fresh bar at 0%.
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
        `${testCount.toLocaleString()} tests · ${accuracy.toFixed(2)}% correct · wrote ${outputPath}`,
      );
      process.stdout.write("\n");
    } else {
      writeProgress(intoBatch / DUMP_EVERY);
    }
  }
};

main();
