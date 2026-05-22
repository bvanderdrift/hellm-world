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

export const modelTrainingHistorySchema = z.object({
  validationLosses: z.array(
    z.object({
      stepIndex: z.number(),
      loss: z.number(),
    }),
  ),
  trainingLosses: z.array(z.number()),
});

export type ModelMetadata = z.infer<typeof modelMetadataSchema>;

export type ModelTrainingHistory = z.infer<typeof modelTrainingHistorySchema>;

export type Model = ModelMetadata & Weights & { history: ModelTrainingHistory };
