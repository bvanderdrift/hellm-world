import { llmForwardPassByTokens } from "./llm.ts";
import { getLatestCheckpointModel } from "../model/model-checkpoint-io.ts";
import { validateModel } from "../model/model-validation.ts";
import { gpuContext } from "../shared/gpu-context.ts";
import {
  benchmark,
  printRow,
  printGpuRow,
  WARMUP_ITERS,
  MEASURE_ITERS,
  type Stats,
} from "../bench-harness.ts";
import { llmForwardPassByTokensOnGPU } from "./llm-gpu.ts";
import { loadWeightsIntoGpu } from "../model/model-gpu-helpers.ts";
import { readFileSync, writeFileSync } from "node:fs";

const MODEL_NAME = "addy";
const TOKEN_COUNTS = [1, 3, 5, 10, 30, 50, 100, 300, 500, 1000];
const CPU_CACHE_PATH = new URL(".llm-bench-cpu-cache.json", import.meta.url)
  .pathname;

const gpuOnly = Bun.argv.includes("--gpu-only");

const loadCpuCache = (): Record<string, Stats> => {
  try {
    return JSON.parse(readFileSync(CPU_CACHE_PATH, "utf8"));
  } catch {
    return {};
  }
};

const saveCpuCache = (cache: Record<string, Stats>) => {
  writeFileSync(CPU_CACHE_PATH, JSON.stringify(cache, null, 2));
};

const main = async () => {
  const model = getLatestCheckpointModel(MODEL_NAME);
  validateModel(model);

  console.log(
    gpuOnly ? "inference GPU-only benchmark" : "inference CPU vs GPU benchmark",
  );
  console.log(`  model=${MODEL_NAME}`);
  console.log(`  warmup=${WARMUP_ITERS}, measure=${MEASURE_ITERS} iters`);

  const cpuCache = loadCpuCache();
  if (gpuOnly && Object.keys(cpuCache).length > 0) {
    console.log("  reusing cached CPU speeds where available");
  }
  console.log("");

  for (const tokenCount of TOKEN_COUNTS) {
    const inputTokens = Array.from(
      { length: tokenCount },
      () => model.vocabulary[0]!,
    );
    const label = `tokens=${tokenCount}  [${model.vocabulary[0]} x${tokenCount}]`;

    let cpuStats: Stats | undefined;
    if (gpuOnly) {
      cpuStats = cpuCache[String(tokenCount)];
    } else {
      cpuStats = await benchmark(() => {
        llmForwardPassByTokens(inputTokens, model, false);
      });
      cpuCache[String(tokenCount)] = cpuStats;
    }

    const weightsBuffer = loadWeightsIntoGpu(model);

    const gpuStats = await benchmark(async () => {
      await llmForwardPassByTokensOnGPU(
        inputTokens,
        model,
        weightsBuffer,
        false,
      );
      await gpuContext.device.queue.onSubmittedWorkDone();
    });

    if (cpuStats) {
      const speedup = cpuStats.median / gpuStats.median;
      printRow(
        label,
        gpuOnly ? "CPU(cache)" : "CPU",
        cpuStats,
        "GPU",
        gpuStats,
        speedup,
      );
    } else {
      printGpuRow(label, gpuStats);
    }
  }

  if (!gpuOnly) saveCpuCache(cpuCache);

  console.log("");
};

if (import.meta.main) {
  await main();
}
