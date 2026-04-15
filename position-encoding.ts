export const getPositionEncoding = (
  tokenCount: number,
  dimensions: number,
): number[][] => {
  return new Array(tokenCount).fill(0).map((_, position) =>
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
};
