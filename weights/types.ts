export interface AttentionHeadWeights {
  Q: number[][];
  K: number[][];
  V: {
    up: number[][];
    down: number[][];
  };
}

export interface AttentionWeights {
  heads: AttentionHeadWeights[];
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
  embeddings: Record<T, number[]>; // C x D
  unembeddings: number[][]; // D x C
  transformers: TransformerWeights[];
};
