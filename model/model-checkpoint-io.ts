import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from "fs";
import { join } from "path";
import { getModelFolderPath, getMetadata } from "./model-io.ts";
import {
  legacyModelTrainingHistorySchema,
  modelTrainingStateSchema,
  type Model,
  type ModelMetadata,
  type ModelTrainingState,
  type TransformerWeights,
  type Weights,
} from "./model-types.ts";
import type { Matrix } from "../shared/matrices/matrices.ts";
import { getModelParameterCount } from "./model-helpers.ts";

const CHECKPOINT_FOLDER_PREFIX = "checkpoint_";
const getCheckpointFolder = (versionNumber: number) =>
  CHECKPOINT_FOLDER_PREFIX + versionNumber.toString().padStart(6, "0");
const CHECKPOINT_TRAINING_STATE_FILE_NAME = "training_state.json";
const CHECKPOINT_WEIGHTS_FILE_NAME = "weights.bin";

export const getCheckpointFolderPath = (
  modelFolderPath: string,
  versionNumber: number,
) => {
  return join(modelFolderPath, getCheckpointFolder(versionNumber));
};

export const getCheckpointModel = (
  modelName: string,
  versionNumber: number,
): Model => {
  const modelFolderPath = getModelFolderPath(modelName);
  const metadata = getMetadata(modelFolderPath);
  const history = getCheckpointTrainingState(modelFolderPath, versionNumber);
  const weights = getCheckpoint(
    metadata,
    join(
      getCheckpointFolderPath(modelFolderPath, versionNumber),
      CHECKPOINT_WEIGHTS_FILE_NAME,
    ),
  );

  return {
    name: modelName,
    ...metadata,
    ...weights,
    trainingState: history,
  };
};

export const getLatestCheckpointModel = (modelName: string): Model => {
  const modelFolderPath = getModelFolderPath(modelName);
  const { versionNumber } = getLatestCheckpointFolderPath(modelFolderPath);

  return getCheckpointModel(modelName, versionNumber);
};

const getCheckpointVersionNumber = (checkpointFolderName: string) => {
  const [_, numberAsStringWithExtension] = checkpointFolderName.split("_");

  const versionNumber = Number(numberAsStringWithExtension);

  if (!Number.isFinite(versionNumber)) {
    throw new Error(
      `Unable to parse checkpoint version number. Input: ${checkpointFolderName} - Output: ${versionNumber}`,
    );
  }

  return versionNumber;
};

export const getLatestCheckpointFolderPath = (modelFolderPath: string) => {
  const checkpointFolders = readdirSync(modelFolderPath).filter((file) =>
    file.startsWith(CHECKPOINT_FOLDER_PREFIX),
  );

  const sortedCheckpoints = checkpointFolders.sort((a, b) =>
    b.localeCompare(a),
  );

  const latestCheckpoint = sortedCheckpoints[0];

  if (!latestCheckpoint) {
    throw new Error(`Failed to find checkpoints in ${modelFolderPath}`);
  }

  return {
    fileName: latestCheckpoint,
    versionNumber: getCheckpointVersionNumber(latestCheckpoint),
  };
};

export const getCheckpointTrainingState = (
  modelFolderPath: string,
  versionNumber: number,
): ModelTrainingState => {
  const currentExpectedFile = join(
    getCheckpointFolderPath(modelFolderPath, versionNumber),
    CHECKPOINT_TRAINING_STATE_FILE_NAME,
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
  const newFolderPath = getCheckpointFolderPath(
    modelFolderPath,
    trainingState.trainingLosses.length,
  );

  mkdirSync(newFolderPath, {
    recursive: true,
  });

  const newFileName = join(newFolderPath, CHECKPOINT_TRAINING_STATE_FILE_NAME);

  writeFileSync(newFileName, JSON.stringify(trainingState));

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
  const modelFolderPath = getModelFolderPath(modelName);

  const newFolderPath = join(
    modelFolderPath,
    getCheckpointFolder(versionNumber),
  );

  mkdirSync(newFolderPath, { recursive: true });

  const flattenedWeights = flattenWeights(weights);

  const newFileName = join(newFolderPath, CHECKPOINT_WEIGHTS_FILE_NAME);

  writeFileSync(newFileName, Buffer.from(flattenedWeights.buffer));

  console.log(`✅ Checkpoint written to ${newFileName}`);
};
