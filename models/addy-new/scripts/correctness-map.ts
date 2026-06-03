/**
 * Correctness testing for the `addy-new` addition model.
 *
 * Given a checkpoint version number it spins up `-m` worker processes (default
 * 1), each of which loads that exact checkpoint and forever samples random
 * pairs (a, b) with a, b < 10_000, running the model on "a+b=" and reporting
 * whether the fully-correct answer is produced. This root process owns the
 * single grid file (`correctness_grid.bin`): it folds every result in and, every
 * 1000 tests, saves the grid and dumps a heatmap PNG via the charting module so
 * accuracy can be inspected visually.
 *
 * The charting half lives in `correctness-chart.ts` and can be run on its own
 * to (re)render an already-saved grid without retesting.
 *
 * This is an independent script: it only imports already-exported inference
 * helpers, it does not modify any inference/training code.
 *
 * Usage: bun run models/addy-new/scripts/correctness-map.ts <checkpointId> [-m <cpus>]
 */

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
import type { WorkerInput, WorkerOutput } from "./correctness-worker.ts";

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

/** Parse `<checkpointId> [-m <cpus>]` from argv; `-m` defaults to 1. */
const parseArgs = (): { checkpointId: number; cpuCount: number } => {
  const args = process.argv.slice(2);
  let checkpointArg: string | undefined;
  let cpuCount = 1;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "-m") {
      cpuCount = Number(args[++i]);
    } else {
      checkpointArg = args[i];
    }
  }

  if (
    checkpointArg === undefined ||
    Number.isNaN(Number(checkpointArg)) ||
    !Number.isInteger(cpuCount) ||
    cpuCount < 1
  ) {
    console.error(
      "Usage: bun run models/addy-new/scripts/correctness-map.ts <checkpointId> [-m <cpus>]",
    );
    process.exit(1);
  }

  return { checkpointId: Number(checkpointArg), cpuCount };
};

const main = async () => {
  const { checkpointId, cpuCount } = parseArgs();

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
    `Loading ${MODEL_NAME} checkpoint ${checkpointId} across ${cpuCount} worker(s)...`,
  );
  console.log("Sampling forever — Ctrl-C to stop.\n");

  // The root owns the grid and persistence; workers only stream results in.
  const onResults = async (results: WorkerOutput["results"]) => {
    for (const { a, b, correct } of results) {
      // A cell may be re-sampled by another worker; only count genuinely new
      // tests so testCount/correctCount stay in sync with the grid's non-null
      // cells (matching how loadGrid derives them).
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

  // Serialize the async result-folding so two batches never interleave their
  // grid writes / chart dumps.
  let queue: Promise<void> = Promise.resolve();

  const workers = new Array(cpuCount).fill(0).map(() => {
    const worker = new Worker(
      "./models/addy-new/scripts/correctness-worker.ts",
    );
    worker.onerror = (event) => {
      console.error("\nWorker error:", event.message ?? event);
      process.exit(1);
    };
    worker.onmessage = (event: MessageEvent<WorkerOutput>) => {
      queue = queue.then(() => onResults(event.data.results));
    };
    const input: WorkerInput = { checkpointId };
    worker.postMessage(input);
    return worker;
  });

  // Keep the process alive forever; workers drive all the work via messages.
  void workers;
};

main();
