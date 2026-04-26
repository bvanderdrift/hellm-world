export const sum = (values: number[]) => {
  return values.reduce((partialSum, e) => e + partialSum, 0);
};

export const dotProduct = (v1: number[], v2: number[]) => {
  if (v1.length !== v2.length) {
    throw new Error(
      `v1 lenght ${v1.length} and v2 length ${v2.length} not overlapping`,
    );
  }

  const multiples = v1.map((e, i) => e * (v2[i] as number));

  return sum(multiples);
};

export const makeLogitsSafeForExponation = (logits: number[]) => {
  const biggestLogit = Math.max(...logits);
  // To prevent overflows. Logits are still related the same since they all subtract the same value
  return {
    safeLogits: logits.map((logit) => logit - biggestLogit),
    biggestLogit,
  };
};

export const softmax = (logits: number[]) => {
  const { safeLogits } = makeLogitsSafeForExponation(logits);

  const sumExponentials = safeLogits.reduce((sum, logit) => {
    const exponential = Math.exp(logit);

    return sum + exponential;
  }, 0);

  return safeLogits.map((logit) => Math.exp(logit) / sumExponentials);
};

/** Rectified Linear Unit */
export const relu = (values: number[]) =>
  values.map((value) => Math.max(value, 0));

export const mean = (values: number[]) => {
  return sum(values) / values.length;
};

export const calculateStandardDeviation = (values: number[]) => {
  const average = mean(values);

  const squareDeltas = values.map((value) => Math.pow(value - average, 2));

  const averageSquareDeltas = mean(squareDeltas);

  return {
    average,
    standardDeviation: Math.sqrt(averageSquareDeltas),
  };
};

export const divideToWhole = (nominator: number, denominator: number) => {
  const divisionRemainder = nominator % denominator;

  if (divisionRemainder !== 0) {
    throw new Error(
      `Can't perfectly divide the nominator ${nominator} by denominator (${denominator})`,
    );
  }

  return Math.round(nominator / denominator);
};
