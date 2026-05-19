import {
  createMatrixBuffer,
  type MatrixBuffer,
} from "../shared/matrices-gpu.ts";
import { getPositionEncoding } from "./position-encoding.ts";

export const getPositionEncodingOnGPU = (
  tokenCount: number,
  dimensions: number,
): MatrixBuffer => {
  const matrix = getPositionEncoding(tokenCount, dimensions);

  return createMatrixBuffer(matrix);
};
