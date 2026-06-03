import { cos, sin } from "typegpu/std";

export const getPositionEncodingOnGPU = (
  i: number,
  j: number,
  dimensions: number,
): number => {
  "use gpu";

  const pairIndex = j - (j % 2);
  const divider = Math.pow(10_000, pairIndex / dimensions);
  const angle = i / divider;

  if (j % 2 === 0) {
    // even
    return sin(angle);
  } else {
    // odd
    return cos(angle);
  }
};
