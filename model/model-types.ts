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

export type ModelMetadata = {
  vocabulary: string[];
  trainingMaskSeparator?: string;
  headsCount: number;
  mlpMultiple: number;
};

export type Model = ModelMetadata & Weights;

export type ModelCheckpoint = {
  // Average loss of every training step, so length is amount of steps taken
  historyLosses: number[];
  weights: Weights;
};
