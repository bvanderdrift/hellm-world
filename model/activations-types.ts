export type AttentionHeadActivations = {
  inputK: number[][];
  inputV: number[][];
  inputQ: number[][];
  attentionRelevancyOutput: number[][];
  softmaxOutput: number[][];
  /** An array of matrixes; one matrix for each embedding vector  */
  lookbackUpdateVectors: number[][][];
  output: number[][];
};

export type AttentionActivations = {
  normalizedInput: number[][];
  heads: AttentionHeadActivations[];
  outMatrixInputActivations: number[][];
  output: number[][];
};

export type MultilayerPerceptronActivations = {
  normalizedInputToUpping: number[][];
  /** Already biased - can reverse-calculate subtracting bias weights */
  uppingToNonLinear: number[][];
  /** C x 4D matrix */
  nonLinearToDowning: number[][];
  /** Already biased - can reverse-calculate subtracting bias weights */
  downingOutput: number[][];
};

export type TransformerActivations = {
  transformerInput: number[][];
  attention: AttentionActivations;
  mlp: MultilayerPerceptronActivations;
  // Can calculate transformer output by taking input and adding both attention + mlp output
};

export type Activations = {
  inputPositionToVocabPosition: number[];
  tokensToPosition: number[][];
  positionToTransformers: number[][];
  transformerActivations: TransformerActivations[];
  transformersToNormalizer: number[][];
  normalizerToUnembeddings: number[][];
  unembeddingsOutputLogits: number[][];
};
