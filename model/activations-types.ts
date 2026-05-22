import type { Matrix } from "../shared/matrices.ts";

export type AttentionHeadActivations = {
  inputK: Matrix;
  inputV: Matrix;
  inputQ: Matrix;
  attentionRelevancyOutput: Matrix;
  softmaxOutput: Matrix;
  output: Matrix;
};

export type AttentionActivations = {
  normalizedInput: Matrix;
  heads: AttentionHeadActivations[];
  outMatrixInputActivations: Matrix;
  output: Matrix;
};

export type MultilayerPerceptronActivations = {
  normalizedInputToUpping: Matrix;
  /** Already biased - can reverse-calculate subtracting bias weights */
  uppingToNonLinear: Matrix;
  /** C x 4D matrix */
  nonLinearToDowning: Matrix;
  /** Already biased - can reverse-calculate subtracting bias weights */
  downingOutput: Matrix;
};

export type TransformerActivations = {
  transformerInput: Matrix;
  attention: AttentionActivations;
  mlp: MultilayerPerceptronActivations;
  // Can calculate transformer output by taking input and adding both attention + mlp output
};

export type Activations = {
  inputPositionToVocabPosition: number[];
  tokensToPosition: Matrix;
  positionToTransformers: Matrix;
  transformerActivations: TransformerActivations[];
  transformersToNormalizer: Matrix;
  normalizerToUnembeddings: Matrix;
  unembeddingsOutputLogits: Matrix;
};
