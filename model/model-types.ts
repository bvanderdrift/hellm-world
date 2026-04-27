export interface AttentionWeights {
  Q: number[][];
  K: number[][];
  V: number[][];
  out: number[][];
}

export interface MultilayerPerceptronWeights {
  wUp: {
    weightsMatrix: number[][];
    biasVector: number[];
  };
  wDown: {
    weightsMatrix: number[][];
    biasVector: number[];
  };
}

export interface TransformerWeights {
  attention: AttentionWeights;
  multilayerPerceptron: MultilayerPerceptronWeights;
}

export type Weights = {
  embeddings: number[][]; // T x D
  unembeddings: number[][]; // D x T
  transformers: TransformerWeights[];
};

export type ModelMetadata = {
  vocabulary: string[];
  headsCount: number;
  mlpMultiple: number;
};

export type Model = ModelMetadata & Weights;

export type ModelCheckpoint = {
  // Average loss of every training step, so length is amount of steps taken
  historyLosses: number[];
  weights: Weights;
};
