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
