export const tokens = ["hello", "world", "my", "name", "is", "beer"] as const;
export type Token = (typeof tokens)[number];

export const tokenize = (input: string): Token[] => {
  let temp = input;

  const matchedTokens: Token[] = [];

  while (temp !== "") {
    const match = tokens.find((t) => temp.startsWith(t));

    if (!match) {
      throw new Error(`Unable to tokenize ${temp}`);
    }

    matchedTokens.push(match);
    temp = temp.replace(match, "").trim();
  }

  return matchedTokens;
};
