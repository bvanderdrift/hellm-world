import { describe, expect, it } from "vitest";
import { tokenize } from "./tokenizer.ts";
import { toyWeights } from "./weights/toy_weights/toyWeights.ts";

describe("tokenize", () => {
  it("tokenizes a known sentence into vocabulary tokens", () => {
    expect(tokenize("hello world my name is beer", toyWeights.tokens)).toEqual([
      "hello",
      "world",
      "my",
      "name",
      "is",
      "beer",
    ]);
  });

  it("handles repeated tokens", () => {
    expect(tokenize("hello hello world", toyWeights.tokens)).toEqual([
      "hello",
      "hello",
      "world",
    ]);
  });

  it("throws when the input contains an unknown token", () => {
    expect(() => tokenize("hello mars", toyWeights.tokens)).toThrow(
      "Unable to tokenize mars",
    );
  });
});
