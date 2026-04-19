export interface AttentionWeights {
  headsCount: number;
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

export type Weights<T extends string = string> = {
  tokens: T[];
  embeddings: Record<T, number[]>; // C x D
  unembeddings: number[][]; // D x C
  transformers: TransformerWeights[];
};
