import {
  createMatrixBuffer,
  type MatrixBuffer,
} from "../shared/matrices-gpu.ts";

export const getPositionEncodingOnGPU = (
  tokenCount: number,
  dimensions: number,
): MatrixBuffer => {
  const matrix = new Array(tokenCount).fill(0).map((_, position) =>
    new Array(dimensions).fill(0).map((_, featureIndex) => {
      const pairIndex = featureIndex - (featureIndex % 2);
      const divider = Math.pow(10_000, pairIndex / dimensions);
      const angle = position / divider;

      if (featureIndex % 2 === 0) {
        // even
        return Math.sin(angle);
      } else {
        // odd
        return Math.cos(angle);
      }
    }),
  );

  return createMatrixBuffer(matrix);
};
