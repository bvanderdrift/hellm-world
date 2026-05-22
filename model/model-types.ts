import z from "zod";
import type { Matrix } from "../shared/matrices.ts";

export interface AttentionWeights {
  Q: Matrix;
  K: Matrix;
  V: Matrix;
  out: Matrix;
}

export interface MultilayerPerceptronWeights {
  wUp: {
    weightsMatrix: Matrix;
    biasVector: Matrix;
  };
  wDown: {
    weightsMatrix: Matrix;
    biasVector: Matrix;
  };
}

export interface TransformerWeights {
  attention: AttentionWeights;
  multilayerPerceptron: MultilayerPerceptronWeights;
}

export type Weights = {
  embeddings: Matrix; // T x D
  unembeddings: Matrix; // D x T
  transformers: TransformerWeights[];
};

export const modelMetadataSchema = z.object({
  vocabulary: z.array(z.string()),
  trainingMaskSeparator: z.string().optional(),
  counts: z.object({
    transformers: z.int().positive(),
    attentionHeads: z.int().positive(),
    hiddenDimensions: z.int().positive(),
    mlpMultiple: z.int().positive(),
  }),
});

export type ModelMetadata = z.infer<typeof modelMetadataSchema>;

export type Model = ModelMetadata & Weights;

export type ModelTrainingHistory = {
  validationLosses: { stepIndex: number; loss: number }[];
  trainingLosses: number[];
};

export type ModelCheckpoint = {
  // Average loss of every training step, so length is amount of steps taken
  history: ModelTrainingHistory;
  weights: Weights;
};
