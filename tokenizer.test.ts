import { describe, expect, it } from "vitest";
import { tokenize } from "./tokenizer.ts";

describe("tokenize", () => {
  const vocab = ["hello", "world", "my", "name", "is", "beer"];

  it("tokenizes a known sentence into vocabulary tokens", () => {
    expect(tokenize("hello world my name is beer", vocab)).toEqual([
      "hello",
      "world",
      "my",
      "name",
      "is",
      "beer",
    ]);
  });

  it("handles repeated tokens", () => {
    expect(tokenize("hello hello world", vocab)).toEqual([
      "hello",
      "hello",
      "world",
    ]);
  });

  it("throws when the input contains an unknown token", () => {
    expect(() => tokenize("hello mars", vocab)).toThrow(
      "Unable to tokenize mars",
    );
  });
});
