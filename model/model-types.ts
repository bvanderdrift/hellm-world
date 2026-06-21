import z from "zod";
import type { Matrix } from "../shared/matrices/matrices.ts";

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

export const legacyModelTrainingHistorySchema = z.object({
  validationLosses: z.array(
    z.object({
      stepIndex: z.number(),
      loss: z.number(),
    }),
  ),
  trainingLosses: z.array(z.number()),
});

export const lossRecordSchema = z.record(
  // coerce b/c JSON will always turn it into a string key
  z.coerce.number(),
  z.number(),
);

export type LossRecord = z.infer<typeof lossRecordSchema>;

export const samplerStateSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("uniform"),
  }),
  z.object({
    type: z.literal("loss-weighted"),
    lossRecord: lossRecordSchema,
  }),
]);

export type SamplerState = z.infer<typeof samplerStateSchema>;

export const modelTrainingStateSchema = legacyModelTrainingHistorySchema.extend(
  {
    samplerState: samplerStateSchema,
  },
);

export type ModelMetadata = z.infer<typeof modelMetadataSchema>;

export type ModelTrainingState = z.infer<typeof modelTrainingStateSchema>;

export type Model = ModelMetadata &
  Weights & { name: string, trainingState: ModelTrainingState };
