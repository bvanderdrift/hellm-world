export const tokenize = <T extends string>(
  input: string,
  vocabTokens: T[],
): T[] => {
  let temp = input;

  const matchedTokens: T[] = [];

  while (temp !== "") {
    const match = vocabTokens.find((t) => temp.startsWith(t));

    if (!match) {
      throw new Error(`Unable to tokenize ${temp}`);
    }

    matchedTokens.push(match);
    temp = temp.replace(match, "").trim();
  }

  return matchedTokens;
};
