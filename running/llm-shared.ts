export const MAX_CONTEXT = 20;

export const pickToken = (
  probabilities: Float32Array,
  vocabulary: string[],
) => {
  const nextTokenIndex = getHighestValueIndex(probabilities);

  const nextToken = vocabulary[nextTokenIndex];

  if (!nextToken) {
    throw new Error(`Failed to find token at index ${nextTokenIndex}`);
  }

  return nextToken;
};

export const getHighestValueIndex = (values: Float32Array) => {
  return values.reduce(
    (tracker, value, index) => {
      if (value > tracker.value) {
        return {
          index,
          value,
        };
      }

      return tracker;
    },
    {
      index: 0,
      value: -Infinity,
    },
  ).index;
};
