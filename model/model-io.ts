import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from "fs";
import { join } from "path";
import type { ModelMetadata, Model, ModelCheckpoint } from "./types.ts";

const METADATA_FILE_NAME = "_metadata.json";
const TRAINING_DATA_FILE_NAME = "_training_data.txt";

export const getLatestCheckpointModel = (
  model: string,
): { historyLosses: number[]; model: Model } => {
  const modelFolderPath = join(import.meta.dirname, model);
  const latestCheckpointFile = getLatestCheckpointFile(modelFolderPath);
  const checkpoint = getCheckpoint(join(modelFolderPath, latestCheckpointFile));

  return {
    historyLosses: checkpoint.historyLosses,
    model: {
      ...getMetadata(join(modelFolderPath, METADATA_FILE_NAME)),
      ...checkpoint.weights,
    },
  };
};

const getLatestCheckpointFile = (modelFolderPath: string): string => {
  const checkpointFiles = readdirSync(modelFolderPath).filter(
    (file) => file !== METADATA_FILE_NAME,
  );

  const sortedCheckpoints = checkpointFiles.sort((a, b) => b.localeCompare(a));

  const latestCheckpoint = sortedCheckpoints[0];

  if (!latestCheckpoint) {
    throw new Error(`Failed to find checkpoints in ${modelFolderPath}`);
  }

  return latestCheckpoint;
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

const getCheckpoint = (pathToCheckpoint: string): ModelCheckpoint => {
  return JSON.parse(readFileSync(pathToCheckpoint).toString());
};

export const readRawTrainingData = (modelName: string) => {
  const modelTrainingDataFile = join(
    import.meta.dirname,
    modelName,
    TRAINING_DATA_FILE_NAME,
  );

  return readFileSync(modelTrainingDataFile).toString();
};

export const writeNewCheckpoint = (
  modelName: string,
  checkpoint: ModelCheckpoint,
) => {
  const modelFolderPath = join(import.meta.dirname, modelName);
  const lastFile = getLatestCheckpointFile(modelFolderPath);

  const [_, numberAsStringWithExtension] = lastFile.split("_");

  const lastNumber = Number(numberAsStringWithExtension?.replace(".json", ""));
  const newNumber = lastNumber + 1;

  writeCheckpoint(modelFolderPath, newNumber, checkpoint);
};

const writeCheckpoint = (
  modelFolderPath: string,
  newNumber: number,
  checkpoint: ModelCheckpoint,
) => {
  const newFileName = `checkpoint_${newNumber.toString().padStart(6, "0")}.json`;

  // layman's pick operation
  const cleanPayload: ModelCheckpoint = {
    historyLosses: checkpoint.historyLosses,
    weights: {
      embeddings: checkpoint.weights.embeddings,
      transformers: checkpoint.weights.transformers,
      unembeddings: checkpoint.weights.unembeddings,
    },
  };

  writeFileSync(
    join(modelFolderPath, newFileName),
    JSON.stringify(cleanPayload),
  );
};

export const writeNewModel = (modelName: string, model: Model) => {
  const modelFolderPath = join(import.meta.dirname, modelName);

  if (existsSync(modelFolderPath)) {
    throw new Error(`Model ${modelName} already has an existing folder`);
  }

  mkdirSync(modelFolderPath);

  const metadata: ModelMetadata = {
    vocabulary: model.vocabulary,
    headsCount: model.headsCount,
    mlpMultiple: model.mlpMultiple,
  };

  writeFileSync(
    join(modelFolderPath, METADATA_FILE_NAME),
    JSON.stringify(metadata, null, 2),
  );

  writeFileSync(
    join(modelFolderPath, TRAINING_DATA_FILE_NAME),
    "", // Initialize empty file
  );

  // First checkpoint file
  writeCheckpoint(modelFolderPath, 0, {
    historyLosses: [],
    weights: model,
  });
};
