import { validateSize } from "./matrices.ts";
import { tokens, type Token } from "./tokenizer.ts";

export const embeddings: Record<Token, number[]> = {
  hello: [1],
  world: [1],
  my: [1],
  name: [1],
  is: [1],
  beer: [1],
};

export const HIDDEN_DIMENSIONS_SIZE = embeddings.beer.length;
export const VOCAB_SIZE = tokens.length;

for (const [token, vector] of Object.entries(embeddings)) {
  if (vector.length !== HIDDEN_DIMENSIONS_SIZE) {
    throw new Error(
      `Token ${token} has unexpected vector length ${vector.length} vs base length ${VOCAB_SIZE}`,
    );
  }
}

export const unembeddingsMatrix: number[][] = [[1]];

validateSize(unembeddingsMatrix, HIDDEN_DIMENSIONS_SIZE);

export const outMatrix: number[][] = [[1, 1, 1, 1, 1, 1]];

validateSize(outMatrix, HIDDEN_DIMENSIONS_SIZE, VOCAB_SIZE);
