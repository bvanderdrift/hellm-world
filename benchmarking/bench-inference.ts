import {
  llmForwardPassByTokens,
  llmForwardPassByTokensOnGPU,
} from "../running/llm.ts";
import { getLatestCheckpointModel } from "../model/model-io.ts";
import { validateModel } from "../model/model-validation.ts";
import { gpuContext } from "../shared/gpu-context.ts";
import { benchmark, printRow, WARMUP_ITERS, MEASURE_ITERS } from "./bench-harness.ts";

const MODEL_NAME = "addy";
const TOKEN_COUNTS = [1, 3, 5, 10];

const main = async () => {
  const { model } = getLatestCheckpointModel(MODEL_NAME);
  validateModel(model);

  console.log("inference CPU vs GPU benchmark");
  console.log(`  model=${MODEL_NAME}`);
  console.log(`  warmup=${WARMUP_ITERS}, measure=${MEASURE_ITERS} iters\n`);

  for (const tokenCount of TOKEN_COUNTS) {
    const inputTokens = model.vocabulary.slice(0, tokenCount);
    const label = `tokens=${tokenCount}  [${inputTokens.join(", ")}]`;

    const cpuStats = await benchmark(() => {
      llmForwardPassByTokens(inputTokens, model, false);
    });

    const gpuStats = await benchmark(async () => {
      await llmForwardPassByTokensOnGPU(inputTokens, model, false);
      await gpuContext.device.queue.onSubmittedWorkDone();
    });

    const speedup = cpuStats.median / gpuStats.median;
    printRow(label, "CPU", cpuStats, "GPU", gpuStats, speedup);
  }

  console.log("");
};

if (import.meta.main) {
  await main();
}
