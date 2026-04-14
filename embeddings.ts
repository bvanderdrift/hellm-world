import type { Token } from "./tokenizer.ts";

export const embeddings: Record<Token, number[]> = {
  beer: [],
  hello: [],
  is: [],
  my: [],
  name: [],
  world: [],
};

const baseLength = embeddings.beer.length;

for (const [token, vector] of Object.entries(embeddings)) {
  if (vector.length !== baseLength) {
    throw new Error(
      `Token ${token} has unexpected vector length ${vector.length} vs base length ${baseLength}`,
    );
  }
}
