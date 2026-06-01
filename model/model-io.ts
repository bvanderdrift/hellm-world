import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import { join } from "path";
import {
  type ModelMetadata,
  type Model,
  modelMetadataSchema,
} from "./model-types.ts";
import { writeTrainingState, writeCheckpoint } from "./model-checkpoint-io.ts";

const METADATA_FILE_NAME = "_metadata.json";
const TRAINING_DATA_FILE_NAME = "_training_data.txt";
const VALIDATION_DATA_FILE_NAME = "_validation_data.txt";

export const getModelFolderPath = (modelName: string) =>
  join(import.meta.dirname, "../models", modelName);

export const getMetadata = (modelFolderPath: string): ModelMetadata => {
  const metadataFilePath = join(modelFolderPath, METADATA_FILE_NAME);
  const metadataJson = readFileSync(metadataFilePath);
const metadata = JSON.parse(metadataJson.toString());

  return modelMetadataSchema.parse(metadata);
};

export const readRawTrainingData = (modelName: string) => {
  const modelTrainingDataFile = join(
    getModelFolderPath(modelName),
    TRAINING_DATA_FILE_NAME,
  );

  return readFileSync(modelTrainingDataFile).toString();
};

export const readRawValidationData = (modelName: string) => {
  const modelValidationDataFile = join(
    getModelFolderPath(modelName),
    VALIDATION_DATA_FILE_NAME,
  );

  return readFileSync(modelValidationDataFile).toString();
};

export const writeNewModel = (modelName: string, model: Model) => {
  const modelFolderPath = getModelFolderPath(modelName);

  if (existsSync(modelFolderPath)) {
    throw new Error(`Model ${modelName} already has an existing folder`);
  }

  mkdirSync(modelFolderPath);

  const metadata: ModelMetadata = {
    vocabulary: model.vocabulary,
    counts: model.counts,
  };

  writeFileSync(
    join(modelFolderPath, METADATA_FILE_NAME),
    JSON.stringify(metadata, null, 2),
  );

  writeFileSync(
    join(modelFolderPath, TRAINING_DATA_FILE_NAME),
    "", // Initialize empty file
  );

  writeTrainingState(modelFolderPath, model.trainingState);

  // First checkpoint file
  writeCheckpoint(modelName, 0, model);
};
