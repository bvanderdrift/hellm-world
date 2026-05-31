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
  modelTrainingStateSchema,
  type ModelTrainingState,
  legacyModelTrainingHistorySchema,
} from "./model-types.ts";
import { getModelParameterCount } from "./model-helpers.ts";
import type { Matrix } from "../shared/matrices.ts";

const METADATA_FILE_NAME = "_metadata.json";
const TRAINING_DATA_FILE_NAME = "_training_data.txt";
const VALIDATION_DATA_FILE_NAME = "_validation_data.txt";

const CHECKPOINT_WEIGHTS_PREFIX = "checkpoint_";
const getCheckpointFileNameNoExtension = (versionNumber: number) =>
  CHECKPOINT_WEIGHTS_PREFIX + versionNumber.toString().padStart(6, "0");
const getCheckpointTrainingDataFileName = (versionNumber: number) =>
  getCheckpointFileNameNoExtension(versionNumber) + "_training_state.json";

export const getModelFolderPath = (modelName: string) =>
  join(import.meta.dirname, modelName);

export const getLatestCheckpointModel = (modelName: string): Model => {
  const modelFolderPath = getModelFolderPath(modelName);
  const { fileName: latestCheckpointFile, versionNumber } =
    getLatestCheckpointFile(modelFolderPath);
  const metadata = getMetadata(modelFolderPath);
  const history = getModelHistory(modelFolderPath, versionNumber);
  const weights = getCheckpoint(
    metadata,
    join(modelFolderPath, latestCheckpointFile),
  );

  return {
    ...metadata,
    ...weights,
    trainingState: history,
  };
};

const getCheckpointVersionNumber = (checkpointFileName: string) => {
  const [_, numberAsStringWithExtension] = checkpointFileName.split("_");

  const versionNumber = Number(
    numberAsStringWithExtension?.replace(".bin", ""),
  );

  if (!Number.isFinite(versionNumber)) {
    throw new Error(
      `Unable to parse checkpoint version number. Input: ${checkpointFileName} - Output: ${versionNumber}`,
    );
  }

  return versionNumber;
};

export const getLatestCheckpointFile = (modelFolderPath: string) => {
  const checkpointFiles = readdirSync(modelFolderPath).filter(
    (file) =>
      file.startsWith(CHECKPOINT_WEIGHTS_PREFIX) && file.endsWith(".bin"),
  );

  const sortedCheckpoints = checkpointFiles.sort((a, b) => b.localeCompare(a));

  const latestCheckpoint = sortedCheckpoints[0];

  if (!latestCheckpoint) {
    throw new Error(`Failed to find checkpoints in ${modelFolderPath}`);
  }

  return {
    fileName: latestCheckpoint,
    versionNumber: getCheckpointVersionNumber(latestCheckpoint),
  };
};

export const getMetadata = (modelFolderPath: string): ModelMetadata => {
  const metadataFilePath = join(modelFolderPath, METADATA_FILE_NAME);
  const metadataJson = readFileSync(metadataFilePath);
  const metadata = JSON.parse(metadataJson.toString());

  return modelMetadataSchema.parse(metadata);
};

export const getModelHistory = (
  modelFolderPath: string,
  versionNumber: number,
): ModelTrainingState => {
  const currentExpectedFile = join(
    modelFolderPath,
    getCheckpointTrainingDataFileName(versionNumber),
  );

  if (existsSync(currentExpectedFile)) {
    return getTrainingState(currentExpectedFile);
  }

  const legacyFile = join(modelFolderPath, "_training_history.json");

  if (!existsSync(legacyFile)) {
    throw new Error(
      `Failed to find either current of legacy training file for checkpoint ${versionNumber}`,
    );
  }

  const historyJson = readFileSync(legacyFile);
  const legacyHistoryJson = JSON.parse(historyJson.toString());
  const legacyHistory =
    legacyModelTrainingHistorySchema.parse(legacyHistoryJson);

  return {
    ...legacyHistory,
    samplerState: {
      type: "loss-weighted",
      lossRecord: {},
    },
  };
};

const getTrainingState = (trainingStateFilePath: string) => {
  const stateJson = readFileSync(trainingStateFilePath);
  const state = JSON.parse(stateJson.toString());

  return modelTrainingStateSchema.parse(state);
};

export const writeTrainingState = (
  modelFolderPath: string,
  trainingState: ModelTrainingState,
) => {
  const newFileName = getCheckpointTrainingDataFileName(
    trainingState.trainingLosses.length,
  );

  writeFileSync(
    join(modelFolderPath, newFileName),
    JSON.stringify(trainingState),
  );

  console.log(`✅ Checkpoint training state written to ${newFileName}`);
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

  const newFileName = getCheckpointFileNameNoExtension(versionNumber) + ".bin";

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
