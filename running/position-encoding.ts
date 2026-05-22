import { createMatrix, getFlatIndex, type Matrix } from "../shared/matrices.ts";

export const getPositionEncoding = (
  tokenCount: number,
  dimensions: number,
): Matrix => {
  const output = createMatrix(tokenCount, dimensions);

  for (let i = 0; i < tokenCount; i++) {
    for (let j = 0; j < dimensions; j++) {
      const pairIndex = j - (j % 2);
      const divider = Math.pow(10_000, pairIndex / dimensions);
      const angle = i / divider;

      if (j % 2 === 0) {
        // even
        output.values[getFlatIndex(i, j, dimensions)] = Math.sin(angle);
      } else {
        // odd
        output.values[getFlatIndex(i, j, dimensions)] = Math.cos(angle);
      }
    }
  }

  return output;
};
