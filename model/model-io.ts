import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from "fs";
import { join } from "path";
import {
  type ModelMetadata,
  type Model,
  modelMetadataSchema,
  type Weights,
  type TransformerWeights,
  modelTrainingHistorySchema,
  type ModelTrainingHistory,
} from "./model-types.ts";
import { getModelParameterCount } from "./model-helpers.ts";
import type { Matrix } from "../shared/matrices.ts";

const METADATA_FILE_NAME = "_metadata.json";
const TRAINING_HISTORY_FILE_NAME = "_training_history.json";
const TRAINING_DATA_FILE_NAME = "_training_data.txt";
const VALIDATION_DATA_FILE_NAME = "_validation_data.txt";

export const getModelFolderPath = (modelName: string) =>
  join(import.meta.dirname, modelName);

export const getLatestCheckpointModel = (modelName: string): Model => {
  const modelFolderPath = getModelFolderPath(modelName);
  const latestCheckpointFile = getLatestCheckpointFile(modelFolderPath);
  const metadata = getMetadata(join(modelFolderPath, METADATA_FILE_NAME));
  const history = getModelHistory(modelFolderPath);
  const weights = getCheckpoint(
    metadata,
    join(modelFolderPath, latestCheckpointFile),
  );

  return {
    ...metadata,
    ...weights,
    history,
  };
};

export const getLatestCheckpointFile = (modelFolderPath: string): string => {
  const checkpointFiles = readdirSync(modelFolderPath).filter(
    (file) => file.startsWith("checkpoint_") && file.endsWith(".bin"),
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

  return modelMetadataSchema.parse(metadata);
};

export const getModelHistory = (modelFolderPath: string) => {
  return getHistory(join(modelFolderPath, TRAINING_HISTORY_FILE_NAME));
};

const getHistory = (historyFilePath: string) => {
  const historyJson = readFileSync(historyFilePath);
  const history = JSON.parse(historyJson.toString());

  return modelTrainingHistorySchema.parse(history);
};

export const writeHistory = (
  modelFolderPath: string,
  history: ModelTrainingHistory,
) => {
  writeFileSync(
    join(modelFolderPath, TRAINING_HISTORY_FILE_NAME),
    JSON.stringify(history),
  );
};

const BYTES_IN_32_BITS = 4;

const getCheckpoint = (
  metadata: ModelMetadata,
  pathToCheckpoint: string,
): Weights => {
  const buffer = readFileSync(pathToCheckpoint);

  return unwrapFlatWeights(metadata, buffer);
};

export const readRawTrainingData = (modelName: string) => {
  const modelTrainingDataFile = join(
    import.meta.dirname,
    modelName,
    TRAINING_DATA_FILE_NAME,
  );

  return readFileSync(modelTrainingDataFile).toString();
};

export const readRawValidationData = (modelName: string) => {
  const modelValidationDataFile = join(
    import.meta.dirname,
    modelName,
    VALIDATION_DATA_FILE_NAME,
  );

  return readFileSync(modelValidationDataFile).toString();
};

export const flattenWeights = (weights: Weights) => {
  const paramCount = getModelParameterCount(weights);
  const singleBuffer = new Float32Array(paramCount);
  let currentOffset = 0;

  const writeToBuffer = (values: Float32Array) => {
    singleBuffer.set(values, currentOffset);
    currentOffset += values.length;
  };

  writeToBuffer(weights.embeddings.values);

  for (const transformer of weights.transformers) {
    writeToBuffer(transformer.attention.K.values);
    writeToBuffer(transformer.attention.V.values);
    writeToBuffer(transformer.attention.Q.values);
    writeToBuffer(transformer.attention.out.values);

    writeToBuffer(transformer.multilayerPerceptron.wUp.weightsMatrix.values);
    writeToBuffer(transformer.multilayerPerceptron.wUp.biasVector.values);
    writeToBuffer(transformer.multilayerPerceptron.wDown.weightsMatrix.values);
    writeToBuffer(transformer.multilayerPerceptron.wDown.biasVector.values);
  }

  writeToBuffer(weights.unembeddings.values);

  if (currentOffset !== paramCount) {
    throw new Error(
      `Unexpected final offset ${currentOffset}. Expected ${paramCount}`,
    );
  }

  return singleBuffer;
};

export const unwrapFlatWeights = (
  metadata: ModelMetadata,
  buffer: NonSharedBuffer,
): Weights => {
  const allValuesFlat = new Float32Array(
    buffer.buffer,
    buffer.byteOffset,
    buffer.byteLength / BYTES_IN_32_BITS,
  );

  let currentOffset = 0;

  const extractMatrix = (vectors: number, dimensions: number): Matrix => {
    const length = vectors * dimensions;

    const m = {
      vectors,
      dimensions,
      values: allValuesFlat.slice(currentOffset, currentOffset + length),
    };

    currentOffset += length;

    return m;
  };

  const {
    vocabulary,
    counts: { hiddenDimensions, mlpMultiple, transformers },
  } = metadata;

  const embeddings = extractMatrix(vocabulary.length, hiddenDimensions);

  const transformerWeights: TransformerWeights[] = [];

  for (
    let transformerIndex = 0;
    transformerIndex < transformers;
    transformerIndex++
  ) {
    const K = extractMatrix(hiddenDimensions, hiddenDimensions);
    const V = extractMatrix(hiddenDimensions, hiddenDimensions);
    const Q = extractMatrix(hiddenDimensions, hiddenDimensions);
    const out = extractMatrix(hiddenDimensions, hiddenDimensions);

    const mUpWeights = extractMatrix(
      hiddenDimensions,
      hiddenDimensions * mlpMultiple,
    );
    const mUpBias = extractMatrix(1, hiddenDimensions * mlpMultiple);

    const mDownWeights = extractMatrix(
      hiddenDimensions * mlpMultiple,
      hiddenDimensions,
    );
    const mDownBias = extractMatrix(1, hiddenDimensions);

    transformerWeights.push({
      attention: {
        K,
        V,
        Q,
        out,
      },
      multilayerPerceptron: {
        wUp: {
          weightsMatrix: mUpWeights,
          biasVector: mUpBias,
        },
        wDown: {
          weightsMatrix: mDownWeights,
          biasVector: mDownBias,
        },
      },
    });
  }

  const unembeddings = extractMatrix(hiddenDimensions, vocabulary.length);

  return {
    embeddings,
    transformers: transformerWeights,
    unembeddings,
  };
};

export const writeCheckpoint = (
  modelName: string,
  versionNumber: number,
  weights: Weights,
) => {
  const modelFolderPath = join(import.meta.dirname, modelName);

  const newFileName = `checkpoint_${versionNumber.toString().padStart(6, "0")}.bin`;

  const flattenedWeights = flattenWeights(weights);

  writeFileSync(
    join(modelFolderPath, newFileName),
    Buffer.from(flattenedWeights.buffer),
  );

  console.log(`✅ Checkpoint written to ${newFileName}`);
};

export const writeNewModel = (modelName: string, model: Model) => {
  const modelFolderPath = join(import.meta.dirname, modelName);

  if (existsSync(modelFolderPath)) {
    throw new Error(`Model ${modelName} already has an existing folder`);
  }

  mkdirSync(modelFolderPath);

  const metadata: ModelMetadata = {
    vocabulary: model.vocabulary,
    counts: model.counts,
  };

  const history: ModelTrainingHistory = {
    trainingLosses: [],
    validationLosses: [],
  };

  writeFileSync(
    join(modelFolderPath, METADATA_FILE_NAME),
    JSON.stringify(metadata, null, 2),
  );

  writeFileSync(
    join(modelFolderPath, TRAINING_DATA_FILE_NAME),
    "", // Initialize empty file
  );

  writeHistory(modelFolderPath, history);

  // First checkpoint file
  writeCheckpoint(modelName, 0, model);
};
