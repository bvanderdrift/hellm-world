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

export type CheckpointWeights = {
  embeddings: number[][]; // T x D
  unembeddings: number[][]; // D x T
  transformers: TransformerWeights[];
};

export type ModelMetadata = {
  vocabulary: string[];
  headsCount: number;
};

export type Weights = ModelMetadata & CheckpointWeights;
