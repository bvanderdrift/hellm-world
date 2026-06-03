/**
 * THIS FILE IS AI-GENERATED
 */

import { writeFileSync } from "fs";
import { join } from "path";
import {
  getCheckpointFolderPath,
  getCheckpointTrainingState,
} from "../model/model-checkpoint-io.ts";
import {
  getMetadata,
  getModelFolderPath,
  readRawTrainingData,
} from "../model/model-io.ts";
import { prepareExampleData } from "../training/prepareExampleData.ts";

const TOP_COUNT = 10;

export const inspectTopLosses = (
  modelName: string,
  checkpointNumber: number,
) => {
  const modelFolderPath = getModelFolderPath(modelName);
  const trainingState = getCheckpointTrainingState(
    modelFolderPath,
    checkpointNumber,
  );

  if (trainingState.samplerState.type !== "loss-weighted") {
    throw new Error(
      `This script only supports the "loss-weighted" sampler, but checkpoint ${checkpointNumber} of model "${modelName}" uses "${trainingState.samplerState.type}". No per-example loss scores are available to inspect.`,
    );
  }

  const { lossRecord } = trainingState.samplerState;

  // Re-derive the exact example ordering the sampler indexes into.
  const metadata = getMetadata(modelFolderPath);

  const trainingData = prepareExampleData(
    readRawTrainingData(modelName),
    metadata.vocabulary,
    metadata.trainingMaskSeparator ?? null,
  );

  // Only examples with an explicitly known loss; ignore the rest.
  const scored = Object.entries(lossRecord)
    .map(([index, loss]) => ({ index: Number(index), loss }))
    .sort((a, b) => b.loss - a.loss)
    .slice(0, TOP_COUNT);

  if (scored.length === 0) {
    console.log(
      `No per-example losses recorded yet for checkpoint ${checkpointNumber} of "${modelName}".`,
    );
    return;
  }

  const lines: string[] = [];
  lines.push(
    `Top ${scored.length} highest-loss examples for "${modelName}" @ checkpoint ${checkpointNumber}:\n`,
  );

  scored.forEach(({ index, loss }, rank) => {
    const example = trainingData[index];
    const text =
      example === undefined
        ? `<no example at index ${index}>`
        : example.sequence.join("");

    lines.push(`#${rank + 1}  loss=${loss.toFixed(6)}  (index ${index})`);
    lines.push(text);
    lines.push("");
  });

  const output = lines.join("\n");
  console.log(output);

  const outputPath = join(
    getCheckpointFolderPath(modelFolderPath, checkpointNumber),
    "struggles.txt",
  );
  writeFileSync(outputPath, output);
  console.log(`Wrote ${outputPath}`);
};
