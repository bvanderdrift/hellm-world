type LossRecord = Map<number, number>;

export const createLossRecord = (): LossRecord => new Map<number, number>();

export const computeSamplingWeights = (
  record: LossRecord,
  totalExamplesCount: number,
) => {
  const losses = Array.from(record.values());

  if (losses.length > totalExamplesCount) {
    throw new Error(`Unexpected losses length: ${losses.length}`);
  }

  const highestLoss = losses.sort((a, b) => b - a)[0] ?? Number.EPSILON;

  return new Array(totalExamplesCount).fill(0).map((_, index) => {
    const storedLoss = record.get(index);

    if (storedLoss === undefined) {
      return highestLoss;
    }

    if (storedLoss === 0) {
      return Number.EPSILON;
    }

    return storedLoss;
  });
};

export const sampleIndices = (weights: number[], count: number) => {
  if (count > weights.length) {
    throw new Error(
      `Requested pick count ${count} exceeds weights count ${weights.length}`,
    );
  }

  let remainingWeights = weights.map((w, i) => ({ w, i }));

  const sampledIndices: number[] = [];

  for (let index = 0; index < count; index++) {
    const summedWeights = remainingWeights.reduce(
      (acc, { w: weight }) => acc + weight,
      0,
    );

    let pointer = Math.random() * summedWeights;

    for (
      let weightIndex = 0;
      weightIndex < remainingWeights.length;
      weightIndex++
    ) {
      const { w: weight, i } = remainingWeights[weightIndex]!;

      if (pointer < 0) {
        continue;
      }

      pointer -= weight;

      if (pointer < 0) {
        sampledIndices.push(i);
        remainingWeights.splice(weightIndex, 1);
      }
    }

    if (sampledIndices.length >= count) {
      break;
    }
  }

  if (sampledIndices.length !== count) {
    throw new Error(
      `Only sampled ${sampledIndices.length} out of expected ${count} indices`,
    );
  }

  return sampledIndices;
};
