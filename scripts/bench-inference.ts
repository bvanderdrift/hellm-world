import { llmForwardPassByTokens } from "../running/llm.ts";
import { getLatestCheckpointModel } from "../model/model-io.ts";
import { validateModel } from "../model/model-validation.ts";

const ITERS = 10;
const MODEL_NAME = "addy";

const { model } = getLatestCheckpointModel(MODEL_NAME);
validateModel(model);

const inputTokens = model.vocabulary.slice(0, 5);

console.log(`inference benchmark`);
console.log(`  model=${MODEL_NAME}  tokens=[${inputTokens.join(", ")}]  iters=${ITERS}\n`);

const samples: number[] = [];

for (let i = 0; i < ITERS; i++) {
  const start = performance.now();
  llmForwardPassByTokens(inputTokens, model, false);
  const elapsed = performance.now() - start;
  samples.push(elapsed);
  console.log(`  run ${i + 1}: ${elapsed.toFixed(3)}ms`);
}

const sorted = [...samples].sort((a, b) => a - b);
const mean = samples.reduce((s, v) => s + v, 0) / samples.length;
const median = sorted[Math.floor(sorted.length / 2)]!;
const min = sorted[0]!;
const max = sorted[sorted.length - 1]!;

console.log(`\n  mean=${mean.toFixed(3)}ms  median=${median.toFixed(3)}ms  min=${min.toFixed(3)}ms  max=${max.toFixed(3)}ms`);
