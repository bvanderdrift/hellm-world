import { readdirSync, readFileSync } from "fs";
import { join } from "path";
import type { CheckpointWeights, ModelMetadata, Weights } from "./types.ts";

const METADATA_FILE_NAME = `_metadata.json`;

export const getLatestCheckpoint = (model: string): Weights => {
  const modelFolderPath = join(import.meta.dirname, model);
  const checkpointFiles = readdirSync(modelFolderPath).filter(
    (file) => file !== METADATA_FILE_NAME,
  );

  const sortedCheckpoints = checkpointFiles.sort((a, b) => b.localeCompare(a));

  const latestCheckpoint = sortedCheckpoints[0];

  if (!latestCheckpoint) {
    throw new Error(`Failed to find checkpoints for ${model}`);
  }

  return {
    ...getMetadata(join(modelFolderPath, METADATA_FILE_NAME)),
    ...getCheckpointWeights(join(modelFolderPath, latestCheckpoint)),
  };
};

const getMetadata = (metadataFilePath: string): ModelMetadata => {
  const metadataJson = readFileSync(metadataFilePath);
  const metadata = JSON.parse(metadataJson.toString());

  if (typeof metadata !== "object" || metadata === null) {
    throw new Error(`Unexpected metadata: ${JSON.stringify(metadata)}`);
  }

  if (!("headsCount" in metadata) || typeof metadata.headsCount !== "number") {
    throw new Error(`Unexpected metadata: ${JSON.stringify(metadata)}`);
  }

  if (
    !("vocabulary" in metadata) ||
    !Array.isArray(metadata.vocabulary) ||
    !metadata.vocabulary.every((token: unknown) => typeof token === "string")
  ) {
    throw new Error(`Unexpected metadata: ${JSON.stringify(metadata)}`);
  }

  return metadata;
};

const getCheckpointWeights = (pathToCheckpoint: string): CheckpointWeights => {
  return JSON.parse(readFileSync(pathToCheckpoint).toString());
};
