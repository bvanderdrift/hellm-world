export const dotProduct = (v1: number[], v2: number[]) => {
  if (v1.length !== v2.length) {
    throw new Error(
      `v1 lenght ${v1.length} and v2 length ${v2.length} not overlapping`,
    );
  }

  const multiples = v1.map((e, i) => e * (v2[i] as number));

  return multiples.reduce((sum, e) => e + sum, 0);
};

export const exp = (x: number) => Math.pow(Math.E, x);
export const ln = (x: number) => Math.log(x);

export const softmax = (logits: number[]) => {
  const biggestLogit = Math.max(...logits);
  // To prevent overflows. Logits are still related the same since they all subtract the same value
  const adjustedLogits = logits.map((logit) => logit - biggestLogit);

  const sumExponentials = adjustedLogits.reduce((sum, logit) => {
    const exponential = exp(logit);

    return sum + exponential;
  }, 0);

  return adjustedLogits.map((logit) => exp(logit) / sumExponentials);
};

/** Rectified Linear Unit */
export const relu = (values: number[]) =>
  values.map((value) => Math.max(value, 0));
