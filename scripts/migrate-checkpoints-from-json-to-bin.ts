import { readFileSync, readdirSync, writeFileSync } from "fs";
import { join } from "path";
import type {
  ModelMetadata,
  ModelTrainingHistory,
  Weights,
} from "../model/model-types.ts";
import { matrixFrom } from "../testing/testing-utils.ts";
import { writeCheckpoint } from "../model/model-io.ts";

interface AttentionWeights {
  Q: number[][];
  K: number[][];
  V: number[][];
  out: number[][];
}

interface MultilayerPerceptronWeights {
  wUp: {
    weightsMatrix: number[][];
    biasVector: number[];
  };
  wDown: {
    weightsMatrix: number[][];
    biasVector: number[];
  };
}

interface TransformerWeights {
  attention: AttentionWeights;
  multilayerPerceptron: MultilayerPerceptronWeights;
}

type OldWeights = {
  embeddings: number[][]; // T x D
  unembeddings: number[][]; // D x T
  transformers: TransformerWeights[];
};

type OldCheckpoint = {
  weights: OldWeights;
  history: ModelTrainingHistory;
};

const migrateOldCheckpoint = (old: OldCheckpoint): Weights => ({
  embeddings: matrixFrom(old.weights.embeddings),
  transformers: old.weights.transformers.map((t) => ({
    multilayerPerceptron: {
      wDown: {
        weightsMatrix: matrixFrom(t.multilayerPerceptron.wDown.weightsMatrix),
        biasVector: matrixFrom([t.multilayerPerceptron.wDown.biasVector]),
      },
      wUp: {
        weightsMatrix: matrixFrom(t.multilayerPerceptron.wUp.weightsMatrix),
        biasVector: matrixFrom([t.multilayerPerceptron.wUp.biasVector]),
      },
    },
    attention: {
      K: matrixFrom(t.attention.K),
      V: matrixFrom(t.attention.V),
      Q: matrixFrom(t.attention.Q),
      out: matrixFrom(t.attention.out),
    },
  })),
  unembeddings: matrixFrom(old.weights.unembeddings),
});

const modelBaseDir = join(import.meta.dirname, "../model");
const modelNames = ["timmy", "addy"];

for (const modelName of modelNames) {
  const modelFolderPath = join(modelBaseDir, modelName);

  const checkpointJsonFiles = readdirSync(modelFolderPath)
    .filter((file) => file.startsWith("checkpoint_") && file.endsWith(".json"))
    .sort((a, b) => a.localeCompare(b));

  if (checkpointJsonFiles.length === 0) {
    console.log(`⏭️ No checkpoint JSON files found in ${modelName}, skipping`);
    continue;
  }

  console.log(
    `\n📦 ${modelName}: found ${checkpointJsonFiles.length} checkpoint JSON files`,
  );

  let lastHistory: ModelTrainingHistory | null = null;

  for (const filename of checkpointJsonFiles) {
    const filePath = join(modelFolderPath, filename);
    const old: OldCheckpoint = JSON.parse(readFileSync(filePath, "utf-8"));

    const weights = migrateOldCheckpoint(old);
    lastHistory = old.history;

    const versionNumber = Number(
      filename.replace("checkpoint_", "").replace(".json", ""),
    );
    writeCheckpoint(modelFolderPath, versionNumber, weights);
  }

  const metadataPath = join(modelFolderPath, "_metadata.json");
  const metadata: ModelMetadata = JSON.parse(
    readFileSync(metadataPath, "utf-8"),
  );
  writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));

  console.log(
    `✅ ${modelName}: updated _metadata.json with history from last checkpoint`,
  );
}
