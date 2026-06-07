/**
 * Correctness benchmark for an OpenAI-compatible endpoint (e.g. vLLM-hosted
 * open-source LLMs), mirroring the `addy-new` correctness map so results can be
 * compared apples-to-apples against our own model.
 *
 * Given a model name (passed straight into the OpenAI client's `model` field),
 * it forever samples random pairs (a, b) with a, b < 10_000, asks the model to
 * compute "a+b=" and records whether the answer is exactly correct. Results are
 * folded into a single grid file (`correctness_grid.bin`) living in a folder
 * named after the model, and every 1000 tests the grid is saved and a heatmap
 * PNG is dumped via the shared charting module.
 *
 * The grid format/encoding is identical to `correctness-grid.ts` (and reuses its
 * constants), so the existing chart script can also read these grids.
 *
 * The endpoint base URL is hardcoded below. The API key is required and comes
 * from `--api-key` or the `OPENAI_API_KEY` env var.
 *
 * This is an independent benchmark script: it does not import or modify any
 * inference/training code.
 *
 * Usage:
 *   bun run models/addy-new/scripts/correctness-map-openai.ts <model> \
 *     [--api-key <key>]
 */

import OpenAI from "openai";

import { GRID_SIZE, TRUE, FALSE } from "./correctness-grid.ts";
import {
  benchmarkFolderPath,
  benchmarkGridPath,
  loadBenchmarkGrid,
  saveBenchmarkGrid,
} from "./correctness-grid-benchmark.ts";
import { renderCorrectnessImage } from "./correctness-chart.ts";

const DUMP_EVERY = 1000;
const PROGRESS_WIDTH = 30;

// Requests are started at this fixed rate regardless of how many are already in
// flight (unlimited concurrency). Raise it to pile more load onto the backend;
// the progress line's in-flight count and last-call duration reveal when the
// GPU can no longer keep up.
const LAUNCH_RATE_PER_SECOND = 500;

// ── progress ──────────────────────────────────────────────────────────────

const writeProgress = (
  fraction: number,
  inFlight: number,
  lastDurationMs: number | null,
) => {
  const filled = Math.round(fraction * PROGRESS_WIDTH);
  const bar = "-".repeat(filled) + " ".repeat(PROGRESS_WIDTH - filled);
  const pct = Math.round(fraction * 100)
    .toString()
    .padStart(3);
  const last =
    lastDurationMs === null ? "—" : `${Math.round(lastDurationMs)}ms`;
  // Trailing spaces clear leftovers when the line shrinks (e.g. in-flight drops).
  process.stdout.write(
    `\r[${bar}] ${pct}% · ${inFlight.toLocaleString()} in flight · last ${last}   `,
  );
};

// ── args ──────────────────────────────────────────────────────────────────

type Args = {
  model: string;
  apiKey: string;
};

const parseArgs = (): Args => {
  const args = process.argv.slice(2);
  let model: string | undefined;
  let apiKey: string | undefined = process.env.OPENAI_API_KEY;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--api-key") {
      apiKey = args[++i];
    } else {
      model = arg;
    }
  }

  if (model === undefined || model.length === 0) {
    console.error(
      "Usage: bun run models/addy-new/scripts/correctness-map-openai.ts <model> [--api-key <key>]",
    );
    process.exit(1);
  }

  if (apiKey === undefined || apiKey.length === 0) {
    console.error(
      "An API key is required: pass --api-key <key> or set OPENAI_API_KEY.",
    );
    process.exit(1);
  }

  return { model, apiKey };
};

// ── correctness test ──────────────────────────────────────────────────────

const SYSTEM_PROMPT =
  "You are a calculator. Respond with only the integer result and nothing else.";

/** Ask the model for a+b and return whether the reply is exactly correct. */
const isCorrect = async (
  client: OpenAI,
  model: string,
  a: number,
  b: number,
): Promise<boolean> => {
  const response = await client.chat.completions.create({
    model,
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: `${a}+${b}=` },
    ],
    temperature: 0,
    max_tokens: 16,
    // Disable reasoning/thinking traces (e.g. Qwen3) via vLLM's chat template so
    // we get only the answer — not in the OpenAI types, hence the cast.
    chat_template_kwargs: { enable_thinking: false },
  } as OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming);

  const reply = response.choices[0]?.message?.content?.trim() ?? "";
  return reply === String(a + b);
};

// ── main ──────────────────────────────────────────────────────────────────

const main = async () => {
  const { model, apiKey } = parseArgs();

  const client = new OpenAI({
    baseURL: "https://hvuq8u925pnz6y-8000.proxy.runpod.net/v1",
    apiKey,
  });

  const {
    grid,
    testCount: resumedTestCount,
    correctCount: resumedCorrectCount,
  } = loadBenchmarkGrid(model);
  let testCount = resumedTestCount;
  let correctCount = resumedCorrectCount;

  if (testCount > 0) {
    console.log(
      `Resuming from ${benchmarkGridPath(model)} — ${testCount.toLocaleString()} cells already tested.`,
    );
  }

  console.log(
    `Benchmarking "${model}" against ${client.baseURL} at ${LAUNCH_RATE_PER_SECOND} req/s...`,
  );
  console.log("Sampling forever — Ctrl-C to stop.\n");

  let inFlight = 0;
  let lastDurationMs: number | null = null;

  // The persist/chart step is async, so serialize it: with unlimited concurrency
  // multiple completions could otherwise overlap their grid writes / chart dumps.
  let dumpQueue: Promise<void> = Promise.resolve();

  // Fold one completed result into the grid + counts (synchronous, hence atomic
  // under JS's single thread) and persist/re-chart every DUMP_EVERY new tests.
  const onResult = (a: number, b: number, correct: boolean) => {
    // A cell may be re-sampled; only count genuinely new tests so the counts
    // stay in sync with the grid's non-null cells (matching loadGrid).
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
      writeProgress(1, inFlight, lastDurationMs);
      process.stdout.write("\n");

      // Snapshot the counts now; they keep advancing while the dump runs.
      const dumpTestCount = testCount;
      const dumpCorrectCount = correctCount;
      dumpQueue = dumpQueue.then(async () => {
        saveBenchmarkGrid(grid, model);
        const outputPath = await renderCorrectnessImage({
          grid,
          outputDir: benchmarkFolderPath(model),
          label: model,
          testCount: dumpTestCount,
          correctCount: dumpCorrectCount,
        });
        const accuracy = (dumpCorrectCount / dumpTestCount) * 100;
        console.log(
          `${dumpTestCount.toLocaleString()} tests · ${accuracy.toFixed(2)}% correct · wrote ${outputPath}`,
        );
        process.stdout.write("\n");
      });
    } else {
      writeProgress(intoBatch / DUMP_EVERY, inFlight, lastDurationMs);
    }
  };

  // Fire one request without awaiting it, so the ticker keeps launching on
  // schedule no matter how long the call takes.
  const launchOne = () => {
    const a = Math.floor(Math.random() * GRID_SIZE);
    const b = Math.floor(Math.random() * GRID_SIZE);

    inFlight++;
    const start = performance.now();

    isCorrect(client, model, a, b)
      .then((correct) => {
        lastDurationMs = performance.now() - start;
        inFlight--;
        onResult(a, b, correct);
      })
      .catch((error) => {
        // Endpoint/network error: don't record a cell, just note it. We do NOT
        // stall the scheduler — the whole point is to keep applying load.
        lastDurationMs = performance.now() - start;
        inFlight--;
        const message = error instanceof Error ? error.message : String(error);
        process.stdout.write(`\nRequest failed (${message})\n`);
      });
  };

  // Self-paced ticker: advance a cursor by the interval each tick so the launch
  // rate stays steady and doesn't drift or bunch up under event-loop pressure.
  // This timer also keeps the process alive.
  const intervalMs = 1000 / LAUNCH_RATE_PER_SECOND;
  let nextLaunchTime = performance.now();
  const tick = () => {
    launchOne();
    nextLaunchTime += intervalMs;
    const delay = Math.max(0, nextLaunchTime - performance.now());
    setTimeout(tick, delay);
  };
  tick();
};

main();
